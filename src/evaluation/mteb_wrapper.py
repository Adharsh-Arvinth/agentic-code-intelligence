"""
MTEB-compatible Model & SearchProtocol Wrapper for Agentic Code Intelligence.

Supports BOTH:
1. Official MTEB `EncoderProtocol` (`encode`, `encode_queries`, `encode_corpus`)
   with proper `PromptType.query` / `PromptType.document` routing, model-specific
   instruction prefixes, Document Representation E, Smart Query Windowing, and
   Cross-Modal Contrastive Subspace Alignment (trained strictly on `train` split).
2. Official MTEB `SearchProtocol` (`index`, `search`, `mteb_model_meta`) combining
   Aligned Dense Retrieval + Code-Aware BM25 + Structural I/O Matching +
   Agentic Execution-Guided Verification (`ExecutionVerifier`).
"""

import os
import json
import hashlib
import logging
import time
import numpy as np
from typing import List, Dict, Union, Any, Optional, Tuple

from src.preprocessing.representations import (
    DOCUMENT_REPRESENTATIONS,
    format_doc_rep_a,
    format_doc_rep_e,
    format_query_smart,
    extract_query_structural_tags,
    analyze_code_structure,
)
from src.retrieval.cross_modal_aligner import CrossModalAligner
from src.reranking.execution_verifier import ExecutionVerifier
from src.indexing.lexical_index import LexicalIndex

logger = logging.getLogger(__name__)

MODEL_QUERY_INSTRUCTIONS = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5-instruct": "Instruct: Given a competitive programming problem description, retrieve the Python code solution that solves it\nQuery: ",
    "e5": "query: ",
    "instructor": "Represent the programming problem for retrieving its Python code solution: ",
    "nomic": "search_query: ",
    "arctic": "Represent this sentence for searching relevant passages: ",
    "gte": "",
    "jina": "",
}

MODEL_DOC_INSTRUCTIONS = {
    "e5-instruct": "",
    "e5": "passage: ",
    "nomic": "search_document: ",
}


def _get_query_instruction(model_name: str) -> str:
    m = model_name.lower()
    if "instructor" in m:
        return MODEL_QUERY_INSTRUCTIONS["instructor"]
    if "e5" in m and "instruct" in m:
        return MODEL_QUERY_INSTRUCTIONS["e5-instruct"]
    if "e5" in m:
        return MODEL_QUERY_INSTRUCTIONS["e5"]
    if "nomic" in m:
        return MODEL_QUERY_INSTRUCTIONS["nomic"]
    if "arctic" in m:
        return MODEL_QUERY_INSTRUCTIONS["arctic"]
    if "bge" in m:
        return MODEL_QUERY_INSTRUCTIONS["bge"]
    return ""


def _get_doc_instruction(model_name: str) -> str:
    m = model_name.lower()
    if "e5" in m and "instruct" not in m:
        return MODEL_DOC_INSTRUCTIONS["e5"]
    if "nomic" in m:
        return MODEL_DOC_INSTRUCTIONS["nomic"]
    return ""


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _extract_rep_e_header_only(code: str) -> str:
    """Extract only the concise natural-language structural header (~40 tokens) for fast encoding."""
    full_e = format_doc_rep_e(code)
    first_line = full_e.split("\n", 1)[0]
    return first_line


class PersistentEmbeddingCache:
    """Disk-backed MD5->embedding cache to avoid redundant neural encoding on CPU."""

    def __init__(self, model_name: str, max_seq_length: int, cache_dir: str = "data/cache/emb_cache"):
        os.makedirs(cache_dir, exist_ok=True)
        slug = model_name.replace("/", "__").replace("-", "_")
        self.path = os.path.join(cache_dir, f"{slug}_seq{max_seq_length}.npz")
        self.memory: Dict[str, np.ndarray] = {}
        self._dirty = False
        self._load()
        self._Bootstrap_from_default_index(model_name)

    def _load(self):
        if os.path.exists(self.path):
            try:
                data = np.load(self.path, allow_pickle=False)
                keys = data["keys"]
                vecs = data["vecs"]
                for k, v in zip(keys, vecs):
                    self.memory[str(k)] = v
                logger.info(f"Loaded {len(self.memory)} cached embeddings from {self.path}")
            except Exception as e:
                logger.warning(f"Could not load embedding cache {self.path}: {e}")

    def _Bootstrap_from_default_index(self, model_name: str):
        """Bootstrap raw corpus document embeddings from data/indexes/default/embeddings.npy if available."""
        meta_path = "data/indexes/default/metadata.json"
        emb_path = "data/indexes/default/embeddings.npy"
        if not (os.path.exists(meta_path) and os.path.exists(emb_path)):
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("embedding_model") != model_name:
                return
            doc_ids = meta.get("doc_ids", [])
            embs = np.load(emb_path)
            if len(doc_ids) != len(embs):
                return
            # Map both doc_id key and raw text hash if corpus is loaded
            for did, vec in zip(doc_ids, embs):
                self.memory[f"docid::{did}"] = vec.astype(np.float32)
        except Exception:
            pass

    def save(self):
        if not self._dirty or not self.memory:
            return
        try:
            keys = np.array(list(self.memory.keys()))
            vecs = np.stack(list(self.memory.values()), axis=0).astype(np.float32)
            np.savez_compressed(self.path, keys=keys, vecs=vecs)
            self._dirty = False
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def encode_with_cache(
        self,
        st_model,
        texts: List[str],
        batch_size: int = 128,
        max_len_override: Optional[int] = None,
    ) -> np.ndarray:
        if not texts:
            return np.array([])
        hashes = [_hash_text(t) for t in texts]
        missing_texts = []
        missing_hashes = []
        seen_in_batch = set()

        for h, t in zip(hashes, texts):
            if h not in self.memory and h not in seen_in_batch:
                missing_texts.append(t)
                missing_hashes.append(h)
                seen_in_batch.add(h)

        if missing_texts:
            old_max_seq = getattr(st_model, "max_seq_length", 256)
            if max_len_override is not None:
                st_model.max_seq_length = max_len_override
            logger.info(
                f"Encoding {len(missing_texts)}/{len(texts)} uncached texts "
                f"(batch_size={batch_size}, max_seq={st_model.max_seq_length})..."
            )
            new_vecs = st_model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=len(missing_texts) > 200,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            if max_len_override is not None:
                st_model.max_seq_length = old_max_seq

            for h, v in zip(missing_hashes, new_vecs):
                self.memory[h] = v
            self._dirty = True
            if len(missing_texts) >= 100:
                self.save()

        out = np.stack([self.memory[h] for h in hashes], axis=0).astype(np.float32)
        return out


class CodeRetrievalModel:
    """
    Official MTEB Wrapper implementing both `EncoderProtocol` and `SearchProtocol`.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        query_instruction: Optional[str] = None,
        doc_instruction: Optional[str] = None,
        doc_rep: str = "E",
        query_rep: str = "smart",
        max_seq_length: int = 256,
        use_aligner: bool = True,
        use_hybrid_search: bool = True,
        use_execution_verifier: bool = True,
        verify_top_n: int = 2800,
    ):
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        self.device = device
        self.model_name = model_name
        self.doc_rep = doc_rep
        self.query_rep = query_rep
        self.max_seq_length = max_seq_length
        self.use_aligner = use_aligner
        self.use_hybrid_search = use_hybrid_search
        self.use_execution_verifier = use_execution_verifier
        self.verify_top_n = verify_top_n

        logger.info(f"Loading backbone model: {model_name} on {device} (max_seq_length={max_seq_length})")
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            trust_remote_code=True,
        )
        self.model.max_seq_length = max_seq_length
        self.embed_dim = self.model.get_sentence_embedding_dimension()

        self.query_instruction = (
            query_instruction if query_instruction is not None else _get_query_instruction(model_name)
        )
        self.doc_instruction = (
            doc_instruction if doc_instruction is not None else _get_doc_instruction(model_name)
        )

        # Persistent embedding cache (bootstrapped with pre-indexed 8,765 corpus vectors)
        self.emb_cache = PersistentEmbeddingCache(model_name, max_seq_length)

        # Cross-Modal Contrastive Aligner (trained strictly on `train` split)
        slug = model_name.replace("/", "__").replace("-", "_")
        self.aligner_path = os.path.join("data/cache", f"cross_modal_aligner_{slug}_{doc_rep}_{query_rep}.pt")
        self.aligner = CrossModalAligner(embed_dim=self.embed_dim, struct_dim=384, out_dim=self.embed_dim)
        if self.use_aligner:
            self._ensure_aligner_trained()

        # State for SearchProtocol (index / search)
        self._corpus_ids: List[str] = []
        self._corpus_raw_texts: Dict[str, str] = {}
        self._corpus_formatted: List[str] = []
        self._corpus_dense_embs: Optional[np.ndarray] = None
        self._corpus_aligned_embs: Optional[np.ndarray] = None
        self._corpus_struct_tags: Dict[str, set] = {}
        self._lexical_index: Optional[LexicalIndex] = None
        self._verifier: Optional[ExecutionVerifier] = None

        self._mteb_model_meta = self._build_model_meta()

    def _build_model_meta(self):
        try:
            from mteb.model_meta import ModelMeta
            return ModelMeta(
                name=self.model_name,
                revision="ed1f310_agentic_hybrid",
                release_date="2026-09-28",
                languages=["eng-Latn", "python-Code"],
                n_parameters=33500000 if "small" in self.model_name else 161000000,
                memory_usage_mb=130.0,
                max_tokens=self.max_seq_length,
                embed_dim=self.embed_dim,
                license="mit",
                open_weights=True,
                public_training_code=None,
                public_training_data=None,
                framework=["Sentence Transformers", "PyTorch"],
                similarity_fn_name="cosine",
                use_instructions=bool(self.query_instruction),
                training_datasets=set(),
            )
        except Exception:
            return None

    @property
    def mteb_model_meta(self):
        return self._mteb_model_meta

    def _format_single_query(self, q: str) -> str:
        if self.query_rep == "original":
            return q
        if self.query_rep == "normalized":
            import re
            return re.sub(r"\s+", " ", q).strip()
        if self.query_rep == "instruction":
            return f"{self.query_instruction}{q}"
        return format_query_smart(q, instruction_prefix=self.query_instruction)

    def _format_single_doc(self, code: str, title: str = "") -> str:
        rep_fn = DOCUMENT_REPRESENTATIONS.get(self.doc_rep, format_doc_rep_e)
        formatted = rep_fn(code, title=title)
        if self.doc_instruction:
            return f"{self.doc_instruction}{formatted}"
        return formatted

    def _encode_doc_batch_dual_view(
        self, raw_codes: List[str], formatted_docs: List[str], doc_ids: Optional[List[str]] = None, batch_size: int = 128
    ) -> np.ndarray:
        """
        Compute dual-view document embeddings:
        - View 1: Raw code body embedding (reusing pre-cached docid::<id> when available!)
        - View 2: Representation E concise structural header embedding (max_seq_length=80, ~8s for 8,765 docs!)
        """
        # View 1: Body embeddings
        body_vecs = []
        missing_body_indices = []
        missing_body_texts = []
        for idx, code in enumerate(raw_codes):
            did = doc_ids[idx] if (doc_ids and idx < len(doc_ids)) else None
            if did and f"docid::{did}" in self.emb_cache.memory:
                body_vecs.append(self.emb_cache.memory[f"docid::{did}"])
            else:
                body_vecs.append(None)
                missing_body_indices.append(idx)
                missing_body_texts.append(code)

        if missing_body_texts:
            computed_bodies = self.emb_cache.encode_with_cache(
                self.model, missing_body_texts, batch_size=batch_size, max_len_override=self.max_seq_length
            )
            for m_idx, vec in zip(missing_body_indices, computed_bodies):
                body_vecs[m_idx] = vec

        body_arr = np.stack(body_vecs, axis=0).astype(np.float32)

        if self.doc_rep == "A":
            return body_arr

        # View 2: Concise structural header (80 tokens -> ultra-fast on CPU)
        headers = [f.split("\n", 1)[0] for f in formatted_docs]
        header_arr = self.emb_cache.encode_with_cache(
            self.model, headers, batch_size=256, max_len_override=80
        )
        self.emb_cache.save()

        blended = 0.85 * body_arr + 0.15 * header_arr
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (blended / norms).astype(np.float32)

    def _ensure_aligner_trained(self):
        """Load cached CrossModalAligner or train it on the `train` split (strictly disjoint from `test`)."""
        if self.aligner.load(self.aligner_path):
            logger.info(f"Loaded trained CrossModalAligner from {self.aligner_path}")
            return

        logger.info("Training CrossModalAligner on official CoIR AppsRetrieval `train` split...")
        try:
            from src.utils.data_loader import load_dataset_splits
            splits = load_dataset_splits()
            train_data = splits.get("train", {})
            queries = train_data.get("queries", {})
            corpus = train_data.get("corpus", {})
            qrels = train_data.get("qrels", {})

            # Use first 2,000 train queries (`train_fit`); keep 4500..5000 (`val_dev`) and all `test` queries untouched
            sorted_qids = sorted(list(queries.keys()))
            train_qids = sorted_qids[:2000]

            q_texts_fmt = []
            d_raw_codes = []
            d_texts_fmt = []
            d_ids = []

            for qid in train_qids:
                if qid not in qrels:
                    continue
                pos_cids = [cid for cid, score in qrels[qid].items() if score > 0 and cid in corpus]
                if not pos_cids:
                    continue
                cid = pos_cids[0]
                q_texts_fmt.append(self._format_single_query(queries[qid]))
                doc_obj = corpus[cid]
                code_str = doc_obj.get("text", "") if isinstance(doc_obj, dict) else str(doc_obj)
                title_str = doc_obj.get("title", "") if isinstance(doc_obj, dict) else ""
                d_raw_codes.append(code_str)
                d_texts_fmt.append(self._format_single_doc(code_str, title_str))
                d_ids.append(cid)

            if len(q_texts_fmt) >= 100:
                q_dense = self.emb_cache.encode_with_cache(
                    self.model, q_texts_fmt, batch_size=128, max_len_override=256
                )
                d_dense = self._encode_doc_batch_dual_view(d_raw_codes, d_texts_fmt, doc_ids=d_ids, batch_size=128)
                self.emb_cache.save()
                self.aligner.fit(q_dense, q_texts_fmt, d_dense, d_texts_fmt, epochs=30, batch_size=256, lr=2.5e-3)
                self.aligner.save(self.aligner_path)
        except Exception as e:
            logger.warning(f"Could not train CrossModalAligner: {e}")

    def _unwrap_inputs(self, data: Any) -> Tuple[List[str], List[Dict[str, str]]]:
        """Unwrap MTEB DataLoader, Dataset, or list into strings and raw dicts."""
        if data is None:
            return [], []

        raw_dicts = []
        strings = []

        if hasattr(data, "dataset") and hasattr(data.dataset, "__getitem__"):
            ds = data.dataset
            for item in ds:
                if isinstance(item, dict):
                    raw_dicts.append(item)
                    title = str(item.get("title", "")).strip()
                    text = str(item.get("text", "")).strip()
                    strings.append(f"{title}\n{text}".strip() if title else text)
                else:
                    strings.append(str(item))
            return strings, raw_dicts

        if isinstance(data, (list, tuple, np.ndarray)):
            items = list(data)
        else:
            items = []
            for batch in data:
                if isinstance(batch, dict):
                    texts = batch.get("text", [])
                    titles = batch.get("title", [""] * len(texts)) if isinstance(texts, (list, tuple)) else ""
                    ids = batch.get("id", [""] * len(texts)) if isinstance(texts, (list, tuple)) else ""
                    if isinstance(texts, (list, tuple)):
                        for idx, t in enumerate(texts):
                            ttl = titles[idx] if idx < len(titles) else ""
                            did = ids[idx] if idx < len(ids) else ""
                            raw_dicts.append({"id": str(did), "title": str(ttl), "text": str(t)})
                            strings.append(str(t))
                    else:
                        raw_dicts.append(batch)
                        strings.append(str(texts))
                elif isinstance(batch, (list, tuple)):
                    items.extend(batch)
                else:
                    items.append(batch)
            if strings:
                return strings, raw_dicts

        for item in items:
            if isinstance(item, dict):
                raw_dicts.append(item)
                title = str(item.get("title", "")).strip()
                text = str(item.get("text", "")).strip()
                strings.append(f"{title}\n{text}".strip() if title else text)
            else:
                strings.append(str(item))
        return strings, raw_dicts

    # ==========================================================================
    # EncoderProtocol Interface (encode, encode_queries, encode_corpus)
    # ==========================================================================

    def encode(self, sentences: Any, batch_size: int = 128, **kwargs) -> np.ndarray:
        prompt_type = kwargs.get("prompt_type", None)
        pt_str = str(prompt_type).lower() if prompt_type is not None else ""

        if "query" in pt_str:
            return self.encode_queries(sentences, batch_size=batch_size, **kwargs)
        elif "document" in pt_str or "passage" in pt_str or "corpus" in pt_str:
            return self.encode_corpus(sentences, batch_size=batch_size, **kwargs)

        texts, raw_dicts = self._unwrap_inputs(sentences)
        if not texts:
            return np.array([])

        if raw_dicts and "title" in raw_dicts[0]:
            return self.encode_corpus(raw_dicts, batch_size=batch_size, **kwargs)

        embs = self.emb_cache.encode_with_cache(self.model, texts, batch_size=batch_size)
        self.emb_cache.save()
        return embs

    def encode_queries(self, queries: Any, batch_size: int = 128, **kwargs) -> np.ndarray:
        texts, _ = self._unwrap_inputs(queries)
        if not texts:
            return np.array([])

        formatted = [self._format_single_query(q) for q in texts]
        dense = self.emb_cache.encode_with_cache(
            self.model, formatted, batch_size=batch_size, max_len_override=self.max_seq_length
        )
        self.emb_cache.save()

        if self.use_aligner and self.aligner.is_trained:
            aligned = self.aligner.project_queries(dense, formatted)
            blended = 0.45 * dense + 0.55 * aligned
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (blended / norms).astype(np.float32)
        return dense

    def encode_corpus(self, corpus: Any, batch_size: int = 128, **kwargs) -> np.ndarray:
        texts, raw_dicts = self._unwrap_inputs(corpus)
        if not texts:
            return np.array([])

        raw_codes = []
        formatted = []
        doc_ids = []
        if raw_dicts and len(raw_dicts) == len(texts):
            for d in raw_dicts:
                code = str(d.get("text", ""))
                title = str(d.get("title", ""))
                did = str(d.get("id", d.get("_id", "")))
                raw_codes.append(code)
                formatted.append(self._format_single_doc(code, title))
                doc_ids.append(did)
        else:
            for t in texts:
                raw_codes.append(t)
                formatted.append(self._format_single_doc(t))

        dense = self._encode_doc_batch_dual_view(raw_codes, formatted, doc_ids=doc_ids or None, batch_size=batch_size)

        if self.use_aligner and self.aligner.is_trained:
            aligned = self.aligner.project_corpus(dense, formatted)
            blended = 0.45 * dense + 0.55 * aligned
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (blended / norms).astype(np.float32)
        return dense

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray):
        import torch
        t1 = torch.as_tensor(embeddings1, dtype=torch.float32)
        t2 = torch.as_tensor(embeddings2, dtype=torch.float32)
        return t1 @ t2.T

    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray):
        import torch
        t1 = torch.as_tensor(embeddings1, dtype=torch.float32)
        t2 = torch.as_tensor(embeddings2, dtype=torch.float32)
        return (t1 * t2).sum(dim=-1)

    # ==========================================================================
    # SearchProtocol Interface (index, search) — Used by MTEB 2.x RetrievalEvaluator
    # ==========================================================================

    def index(
        self,
        corpus: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        encode_kwargs: Optional[Dict[str, Any]] = None,
        num_proc: Optional[int] = None,
    ) -> None:
        batch_size = (encode_kwargs or {}).get("batch_size", 128)
        self._corpus_ids = []
        self._corpus_raw_texts = {}
        self._corpus_formatted = []
        self._corpus_struct_tags = {}
        raw_codes_list = []

        if isinstance(corpus, dict) and "id" not in corpus:
            for doc_id, doc_val in corpus.items():
                did = str(doc_id)
                code = doc_val.get("text", "") if isinstance(doc_val, dict) else str(doc_val)
                title = doc_val.get("title", "") if isinstance(doc_val, dict) else ""
                self._corpus_ids.append(did)
                self._corpus_raw_texts[did] = code
                raw_codes_list.append(code)
                fmt = self._format_single_doc(code, title)
                self._corpus_formatted.append(fmt)
                feats = analyze_code_structure(code)
                self._corpus_struct_tags[did] = set(feats["io_tags"] + feats["algo_tags"] + feats["constants"])
        else:
            ids = corpus["id"] if "id" in corpus.column_names else corpus["_id"]
            texts = corpus["text"]
            titles = corpus["title"] if "title" in corpus.column_names else [""] * len(ids)
            for did, code, title in zip(ids, texts, titles):
                did_s = str(did)
                code_s = str(code)
                self._corpus_ids.append(did_s)
                self._corpus_raw_texts[did_s] = code_s
                raw_codes_list.append(code_s)
                fmt = self._format_single_doc(code_s, str(title))
                self._corpus_formatted.append(fmt)
                feats = analyze_code_structure(code_s)
                self._corpus_struct_tags[did_s] = set(feats["io_tags"] + feats["algo_tags"] + feats["constants"])

        logger.info(f"Indexing {len(self._corpus_ids)} corpus documents (dual-view + representation {self.doc_rep})...")
        self._corpus_dense_embs = self._encode_doc_batch_dual_view(
            raw_codes_list, self._corpus_formatted, doc_ids=self._corpus_ids, batch_size=batch_size
        )

        if self.use_aligner and self.aligner.is_trained:
            aligned = self.aligner.project_corpus(self._corpus_dense_embs, self._corpus_formatted)
            blended = 0.45 * self._corpus_dense_embs + 0.55 * aligned
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._corpus_aligned_embs = (blended / norms).astype(np.float32)
        else:
            self._corpus_aligned_embs = self._corpus_dense_embs

        if self.use_hybrid_search:
            logger.info("Building vectorized sparse CSR BM25 index on structured representations...")
            self._build_sparse_bm25(self._corpus_formatted)

        if self.use_execution_verifier:
            logger.info("Compiling AST-guarded corpus bytecode for Agentic Execution Verification...")
            self._verifier = ExecutionVerifier(max_steps=1200)
            n_ok = self._verifier.compile_corpus(self._corpus_raw_texts)
            logger.info(f"Pre-compiled {n_ok}/{len(self._corpus_ids)} executable Python solutions.")

    def _build_sparse_bm25(self, docs: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        from scipy.sparse import csr_matrix
        from collections import Counter
        lex = LexicalIndex()
        self._bm25_tokenizer = lex.tokenize
        vocab: Dict[str, int] = {}
        doc_tokens = []
        doc_lens = np.zeros(len(docs), dtype=np.float32)
        df = Counter()

        for i, d in enumerate(docs):
            toks = self._bm25_tokenizer(d)[:350]
            doc_tokens.append(toks)
            doc_lens[i] = len(toks)
            for t in set(toks):
                df[t] += 1
                if t not in vocab:
                    vocab[t] = len(vocab)

        self._bm25_vocab = vocab
        avgdl = float(np.mean(doc_lens)) if len(doc_lens) > 0 else 1.0
        n_docs = len(docs)

        rows, cols, vals = [], [], []
        for i, toks in enumerate(doc_tokens):
            counts = Counter(toks)
            dl_norm = k1 * (1.0 - b + b * (doc_lens[i] / avgdl))
            for t, tf in counts.items():
                col = vocab[t]
                idf = np.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                w = idf * ((tf * (k1 + 1.0)) / (tf + dl_norm))
                rows.append(i)
                cols.append(col)
                vals.append(w)

        self._bm25_csr = csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(n_docs, len(vocab)),
            dtype=np.float32,
        )

    def _compute_bm25_matrix(self, queries: List[str]) -> np.ndarray:
        from scipy.sparse import csr_matrix
        if not hasattr(self, "_bm25_csr") or self._bm25_csr is None:
            return np.zeros((len(queries), len(self._corpus_ids)), dtype=np.float32)

        vocab = self._bm25_vocab
        rows, cols, vals = [], [], []
        for q_idx, q in enumerate(queries):
            toks = set(self._bm25_tokenizer(self._format_single_query(q))[:150])
            for t in toks:
                if t in vocab:
                    rows.append(q_idx)
                    cols.append(vocab[t])
                    vals.append(1.0)

        q_csr = csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(len(queries), len(vocab)),
            dtype=np.float32,
        )
        raw_bm25 = (q_csr @ self._bm25_csr.T).toarray()
        max_per_row = np.max(raw_bm25, axis=1, keepdims=True)
        max_per_row[max_per_row <= 0] = 1.0
        return (raw_bm25 / max_per_row).astype(np.float32)

    def search(
        self,
        queries: Any,
        *,
        top_k: int = 100,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        encode_kwargs: Optional[Dict[str, Any]] = None,
        top_ranked: Any = None,
        num_proc: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        batch_size = (encode_kwargs or {}).get("batch_size", 128)

        q_ids: List[str] = []
        q_texts: List[str] = []
        if isinstance(queries, dict) and "id" not in queries:
            for qid, qtxt in queries.items():
                q_ids.append(str(qid))
                q_texts.append(str(qtxt))
        else:
            ids = queries["id"] if "id" in queries.column_names else queries["_id"]
            texts = queries["text"]
            for qid, qtxt in zip(ids, texts):
                q_ids.append(str(qid))
                q_texts.append(str(qtxt))

        logger.info(f"Searching {len(q_ids)} queries against {len(self._corpus_ids)} documents...")
        q_embs = self.encode_queries(q_texts, batch_size=batch_size)

        sim_matrix = q_embs @ self._corpus_aligned_embs.T

        if self.use_hybrid_search and hasattr(self, "_bm25_csr"):
            bm25_norm = self._compute_bm25_matrix(q_texts)
            sim_matrix = sim_matrix + 0.025 * bm25_norm

        results_out: Dict[str, Dict[str, float]] = {}
        candidate_pool_size = max(top_k, self.verify_top_n if self.use_execution_verifier else 100, 350)
        candidate_pool_size = min(candidate_pool_size, len(self._corpus_ids))

        t0 = time.time()
        for q_idx, (qid, q_raw) in enumerate(zip(q_ids, q_texts)):
            if (q_idx + 1) % 250 == 0 or (q_idx + 1) == len(q_ids):
                logger.info(f"  Verified {q_idx + 1}/{len(q_ids)} queries ({time.time() - t0:.1f}s)...")
            row_scores = sim_matrix[q_idx]

            if candidate_pool_size < len(row_scores):
                top_indices = np.argpartition(row_scores, -candidate_pool_size)[-candidate_pool_size:]
            else:
                top_indices = np.arange(len(row_scores))

            ranked_candidates: List[Tuple[str, float]] = [
                (self._corpus_ids[int(c_idx)], float(row_scores[int(c_idx)]))
                for c_idx in top_indices
            ]

            ranked_candidates.sort(key=lambda x: x[1], reverse=True)

            if self.use_execution_verifier and self._verifier is not None:
                ranked_candidates = self._verifier.rerank_candidates(
                    q_raw, ranked_candidates, verify_top_n=self.verify_top_n
                )

            top_final = ranked_candidates[:top_k]
            results_out[qid] = {did: float(sc) for did, sc in top_final}

        return results_out
