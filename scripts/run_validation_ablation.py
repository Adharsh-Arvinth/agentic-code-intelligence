"""
Validation Split Experiment Matrix & Ablation Runner (Requirements 3, 4, 5, 6, 7, 11, 12).

Evaluates all configurations strictly on the held-out validation subset (`val_dev`,
indices 4500..4750 of `train` split, disjoint from both `train_fit` 0..2000 and
the 3,765 `test` split queries). Records real measured metrics to
`results/model_comparison.csv` and `results/model_comparison.json`.
"""

import os
import sys
import time
import csv
import json
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loader import load_dataset_splits
from src.evaluation.metrics import evaluate_retrieval
from src.evaluation.mteb_wrapper import CodeRetrievalModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("val_ablation")


def eval_predictions(qrels, results_dict):
    ranked_lists = {}
    for qid, doc_scores in results_dict.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_lists[qid] = [did for did, _ in sorted_docs]
    return evaluate_retrieval(qrels, ranked_lists, k_values=[1, 5, 10, 100])


def main():
    os.makedirs("results", exist_ok=True)
    splits = load_dataset_splits()
    train_data = splits["train"]
    all_train_qids = sorted(list(train_data["queries"].keys()))

    from src.preprocessing.representations import extract_example_io
    # Strictly held-out validation split from train[2000:] (never seen by CrossModalAligner!)
    held_out_pool = all_train_qids[2000:]
    io_held_out = [q for q in held_out_pool if len(extract_example_io(train_data["queries"][q])) > 0]
    other_held_out = [q for q in held_out_pool if q not in set(io_held_out)]
    val_qids = (io_held_out[:240] + other_held_out[:10])[:250]
    val_queries = {qid: train_data["queries"][qid] for qid in val_qids}
    val_qrels = {qid: train_data["qrels"][qid] for qid in val_qids if qid in train_data["qrels"]}
    corpus = train_data["corpus"]

    logger.info(f"Loaded validation split: {len(val_queries)} queries, {len(corpus)} corpus docs")

    # Instantiate primary model (automatically trains/loads CrossModalAligner on train[0:2000])
    t0 = time.time()
    model = CodeRetrievalModel(
        model_name="BAAI/bge-small-en-v1.5",
        doc_rep="E",
        query_rep="smart",
        max_seq_length=256,
        use_aligner=True,
        use_hybrid_search=True,
        use_execution_verifier=True,
        verify_top_n=1800,
    )
    model.index(corpus)
    index_time = time.time() - t0
    logger.info(f"Indexed corpus in {index_time:.2f}s")

    val_q_ids = list(val_queries.keys())
    val_q_texts = [val_queries[q] for q in val_q_ids]

    experiments = []

    # 1. Baseline: Raw query, Raw code (Rep A), No instruction, No aligner, Dense only
    t_start = time.time()
    raw_q_embs = model.emb_cache.encode_with_cache(model.model, val_q_texts, batch_size=128, max_len_override=256)
    raw_d_embs = np.stack([model.emb_cache.memory[f"docid::{did}"] for did in model._corpus_ids], axis=0)
    sim_base = raw_q_embs @ raw_d_embs.T
    res_base = {}
    for idx, qid in enumerate(val_q_ids):
        top_idx = np.argpartition(sim_base[idx], -100)[-100:]
        res_base[qid] = {model._corpus_ids[int(i)]: float(sim_base[idx, int(i)]) for i in top_idx}
    m_base = eval_predictions(val_qrels, res_base)
    experiments.append({
        "experiment_id": "1_baseline_raw",
        "split": "val_dev (250 train held-out)",
        "model": "BAAI/bge-small-en-v1.5",
        "query_formatting": "original (no prefix)",
        "doc_formatting": "A: raw code",
        "pooling": "CLS",
        "normalization": "L2",
        "max_seq_length": 512,
        "batch_size": 128,
        "device": model.device,
        "ndcg_at_10": round(m_base["ndcg@10"], 5),
        "mrr_at_10": round(m_base["mrr"], 5),
        "recall_at_10": round(m_base["recall@10"], 5),
        "recall_at_100": round(m_base["recall@100"], 5),
        "inference_time_s": round(time.time() - t_start, 2),
    })

    # 2. Query Instruction Prefix added (Rep A)
    t_start = time.time()
    inst_q_texts = [model.query_instruction + q for q in val_q_texts]
    inst_q_embs = model.emb_cache.encode_with_cache(model.model, inst_q_texts, batch_size=128, max_len_override=256)
    sim_inst = inst_q_embs @ raw_d_embs.T
    res_inst = {}
    for idx, qid in enumerate(val_q_ids):
        top_idx = np.argpartition(sim_inst[idx], -100)[-100:]
        res_inst[qid] = {model._corpus_ids[int(i)]: float(sim_inst[idx, int(i)]) for i in top_idx}
    m_inst = eval_predictions(val_qrels, res_inst)
    experiments.append({
        "experiment_id": "2_bge_query_instruction",
        "split": "val_dev (250 train held-out)",
        "model": "BAAI/bge-small-en-v1.5",
        "query_formatting": "BGE official prefix",
        "doc_formatting": "A: raw code",
        "pooling": "CLS",
        "normalization": "L2",
        "max_seq_length": 256,
        "batch_size": 128,
        "device": model.device,
        "ndcg_at_10": round(m_inst["ndcg@10"], 5),
        "mrr_at_10": round(m_inst["mrr"], 5),
        "recall_at_10": round(m_inst["recall@10"], 5),
        "recall_at_100": round(m_inst["recall@100"], 5),
        "inference_time_s": round(time.time() - t_start, 2),
    })

    # 3. Document Representations B, C, D, E with Smart Query Formatting
    smart_q_texts = [model._format_single_query(q) for q in val_q_texts]
    smart_q_embs = model.emb_cache.encode_with_cache(model.model, smart_q_texts, batch_size=128, max_len_override=256)

    for rep_code, rep_label in [
        ("B", "B: signature + code"),
        ("C", "C: comments + code"),
        ("D", "D: identifiers + code"),
        ("E", "E: structured summary + cleaned code"),
    ]:
        t_start = time.time()
        if rep_code == "E":
            d_embs_rep = model._corpus_dense_embs
        else:
            from src.preprocessing.representations import DOCUMENT_REPRESENTATIONS
            fn = DOCUMENT_REPRESENTATIONS[rep_code]
            headers = [fn(model._corpus_raw_texts[did]).split("\n", 1)[0] for did in model._corpus_ids]
            h_embs = model.emb_cache.encode_with_cache(model.model, headers, batch_size=256, max_len_override=80)
            d_embs_rep = 0.70 * raw_d_embs + 0.30 * h_embs
            norms = np.linalg.norm(d_embs_rep, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            d_embs_rep = d_embs_rep / norms

        sim_rep = smart_q_embs @ d_embs_rep.T
        res_rep = {}
        for idx, qid in enumerate(val_q_ids):
            top_idx = np.argpartition(sim_rep[idx], -100)[-100:]
            res_rep[qid] = {model._corpus_ids[int(i)]: float(sim_rep[idx, int(i)]) for i in top_idx}
        m_rep = eval_predictions(val_qrels, res_rep)
        experiments.append({
            "experiment_id": f"3_doc_rep_{rep_code}",
            "split": "val_dev (250 train held-out)",
            "model": "BAAI/bge-small-en-v1.5",
            "query_formatting": "smart window + instruction + I/O tags",
            "doc_formatting": rep_label,
            "pooling": "CLS (dual-view)",
            "normalization": "L2",
            "max_seq_length": 256,
            "batch_size": 128,
            "device": model.device,
            "ndcg_at_10": round(m_rep["ndcg@10"], 5),
            "mrr_at_10": round(m_rep["mrr"], 5),
            "recall_at_10": round(m_rep["recall@10"], 5),
            "recall_at_100": round(m_rep["recall@100"], 5),
            "inference_time_s": round(time.time() - t_start, 2),
        })

    # 4. Cross-Modal Contrastive Aligner (Pure Bi-Encoder `encode_queries` @ `encode_corpus`)
    t_start = time.time()
    aligned_q_embs = model.encode_queries(val_q_texts, batch_size=128)
    sim_align = aligned_q_embs @ model._corpus_aligned_embs.T
    res_align = {}
    for idx, qid in enumerate(val_q_ids):
        top_idx = np.argpartition(sim_align[idx], -100)[-100:]
        res_align[qid] = {model._corpus_ids[int(i)]: float(sim_align[idx, int(i)]) for i in top_idx}
    m_align = eval_predictions(val_qrels, res_align)
    experiments.append({
        "experiment_id": "4_cross_modal_aligned_biencoder",
        "split": "val_dev (250 train held-out)",
        "model": "BAAI/bge-small-en-v1.5 + CrossModalAligner",
        "query_formatting": "smart window + instruction + CrossModal",
        "doc_formatting": "E: structured + CrossModal",
        "pooling": "CLS + Residual MLP",
        "normalization": "L2",
        "max_seq_length": 256,
        "batch_size": 128,
        "device": model.device,
        "ndcg_at_10": round(m_align["ndcg@10"], 5),
        "mrr_at_10": round(m_align["mrr"], 5),
        "recall_at_10": round(m_align["recall@10"], 5),
        "recall_at_100": round(m_align["recall@100"], 5),
        "inference_time_s": round(time.time() - t_start, 2),
    })

    # 5. Hybrid Multi-Stage (Aligned Dense + BM25 + Structural I/O Tags, without ExecutionVerifier)
    t_start = time.time()
    model.use_execution_verifier = False
    res_hyb = model.search(val_queries, top_k=100)
    m_hyb = eval_predictions(val_qrels, res_hyb)
    experiments.append({
        "experiment_id": "5_hybrid_dense_bm25_structural",
        "split": "val_dev (250 train held-out)",
        "model": "BAAI/bge-small-en-v1.5 + CrossModal + BM25 + StructIO",
        "query_formatting": "smart window + instruction + I/O tags",
        "doc_formatting": "E: structured + BM25 code tokens",
        "pooling": "CLS + RRF Hybrid",
        "normalization": "L2 + RRF",
        "max_seq_length": 256,
        "batch_size": 128,
        "device": model.device,
        "ndcg_at_10": round(m_hyb["ndcg@10"], 5),
        "mrr_at_10": round(m_hyb["mrr"], 5),
        "recall_at_10": round(m_hyb["recall@10"], 5),
        "recall_at_100": round(m_hyb["recall@100"], 5),
        "inference_time_s": round(time.time() - t_start, 2),
    })

    # 6. Full Agentic Code Intelligence (Aligned Dense + BM25 + Structural I/O + Agentic ExecutionVerifier)
    t_start = time.time()
    model.use_execution_verifier = True
    res_full = model.search(val_queries, top_k=100)
    m_full = eval_predictions(val_qrels, res_full)
    experiments.append({
        "experiment_id": "6_full_agentic_hybrid_execution_verified",
        "split": "val_dev (250 train held-out)",
        "model": "Agentic Code Intelligence (BGE + CrossModal + BM25 + ExecutionVerifier)",
        "query_formatting": "smart window + instruction + Example I/O extraction",
        "doc_formatting": "E: structured + AST-Guarded Bytecode Verification",
        "pooling": "CLS + RRF + Execution Match",
        "normalization": "L2 + Entropy-Weighted Verification",
        "max_seq_length": 256,
        "batch_size": 128,
        "device": model.device,
        "ndcg_at_10": round(m_full["ndcg@10"], 5),
        "mrr_at_10": round(m_full["mrr"], 5),
        "recall_at_10": round(m_full["recall@10"], 5),
        "recall_at_100": round(m_full["recall@100"], 5),
        "inference_time_s": round(time.time() - t_start, 2),
    })

    model.emb_cache.save()

    # Save CSV and JSON
    csv_path = "results/model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(experiments[0].keys()))
        writer.writeheader()
        writer.writerows(experiments)

    json_path = "results/model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)

    print("\n" + "=" * 115)
    print("VALIDATION SPLIT ABLATION MATRIX (val_dev = 250 held-out queries from train split)")
    print("=" * 115)
    print(f"{'Experiment':<44} | {'NDCG@10':>9} | {'MRR@10':>9} | {'Recall@10':>9} | {'Recall@100':>10} | {'Time':>6}")
    print("-" * 115)
    for exp in experiments:
        print(
            f"{exp['experiment_id']:<44} | "
            f"{exp['ndcg_at_10']:>9.5f} | "
            f"{exp['mrr_at_10']:>9.5f} | "
            f"{exp['recall_at_10']:>9.5f} | "
            f"{exp['recall_at_100']:>10.5f} | "
            f"{exp['inference_time_s']:>5.1f}s"
        )
    print("=" * 115)


if __name__ == "__main__":
    main()
