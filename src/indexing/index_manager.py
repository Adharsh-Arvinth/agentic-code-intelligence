"""Manage embedding and lexical indexes for the code corpus."""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any

from src.indexing.embedding_index import EmbeddingIndex
from src.indexing.lexical_index import LexicalIndex

logger = logging.getLogger(__name__)


class IndexManager:
    """Orchestrate building, loading, and managing embedding and lexical indexes."""
    
    def __init__(self, config):
        """
        Args:
            config: Config dataclass or dict with index configuration.
        """
        self.config = config
        # Support both dataclass and dict config
        self.base_dir = getattr(config, 'index_dir', None) or (config.get('index_dir', 'data/indexes') if isinstance(config, dict) else 'data/indexes')
        os.makedirs(self.base_dir, exist_ok=True)
        self.embedding_index: Optional[EmbeddingIndex] = None
        self.lexical_index: Optional[LexicalIndex] = None
        self.metadata: Optional[Dict] = None
    
    def _get_config_val(self, key: str, default=None):
        """Get config value supporting both dataclass and dict."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def get_index_path(self, version: str, index_type: str) -> str:
        """Get the file path for a specific index type and version."""
        version_dir = os.path.join(self.base_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        paths = {
            'faiss': 'index.faiss',
            'embeddings': 'embeddings.npy',
            'lexical': 'lexical.pkl',
            'metadata': 'metadata.json'
        }
        return os.path.join(version_dir, paths.get(index_type, index_type))

    def is_indexed(self, version: str) -> bool:
        """Check if indexes exist for a given version."""
        return (
            os.path.exists(self.get_index_path(version, 'faiss')) and
            os.path.exists(self.get_index_path(version, 'lexical')) and
            os.path.exists(self.get_index_path(version, 'metadata'))
        )

    def list_versions(self) -> List[str]:
        """List all indexed versions."""
        if not os.path.exists(self.base_dir):
            return []
        return [d for d in os.listdir(self.base_dir) 
                if os.path.isdir(os.path.join(self.base_dir, d))]

    def _extract_texts(self, corpus: Dict[str, Any]) -> tuple:
        """
        Extract document IDs and structured representation E texts from corpus.
        Corpus values can be dicts with 'title'/'text' keys or plain strings.
        """
        from src.preprocessing.representations import format_doc_rep_e
        doc_ids = []
        doc_texts = []
        for doc_id, doc in corpus.items():
            doc_ids.append(doc_id)
            if isinstance(doc, dict):
                title = doc.get('title', '').strip()
                text = doc.get('text', '').strip()
                doc_texts.append(format_doc_rep_e(text, title=title))
            else:
                doc_texts.append(format_doc_rep_e(str(doc)))
        return doc_ids, doc_texts

    def build_all(self, corpus: Dict[str, Any], version: str = 'default'):
        """Build all indexes if they don't already exist."""
        if self.is_indexed(version):
            logger.info(f"Indexes already exist for version '{version}'. Loading...")
            self.load_all(version)
            return
        self.rebuild(corpus, version)

    def rebuild(self, corpus: Dict[str, Any], version: str = 'default'):
        """Force rebuild all indexes."""
        logger.info(f"Building indexes for version '{version}'...")
        doc_ids, doc_texts = self._extract_texts(corpus)
        logger.info(f"Corpus: {len(doc_ids)} documents")
        
        # Build embedding index
        model_name = self._get_config_val('embedding_model', 'jinaai/jina-embeddings-v2-base-code')
        device = self._get_config_val('device', 'cpu')
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        cache_dir = self._get_config_val('cache_dir', 'data/cache')
        batch_size = self._get_config_val('batch_size', 64)
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        start = time.time()
        self.embedding_index = EmbeddingIndex(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir
        )
        
        logger.info(f"Encoding {len(doc_texts)} documents...")
        embeddings = self.embedding_index.encode_documents(
            doc_texts, batch_size=batch_size, show_progress=True
        )
        self.embedding_index.build_index(embeddings)
        self.embedding_index.save(self.get_index_path(version, 'faiss'))
        self.embedding_index.save_embeddings(embeddings, self.get_index_path(version, 'embeddings'))
        embed_time = time.time() - start
        logger.info(f"Embedding index built in {embed_time:.2f}s")
        
        # Build lexical index
        logger.info("Building BM25 lexical index...")
        start = time.time()
        self.lexical_index = LexicalIndex()
        self.lexical_index.build_index(doc_texts, doc_ids)
        self.lexical_index.save(self.get_index_path(version, 'lexical'))
        lex_time = time.time() - start
        logger.info(f"Lexical index built in {lex_time:.2f}s")
        
        # Save metadata
        self.metadata = {
            'version': version,
            'num_documents': len(doc_ids),
            'doc_ids': doc_ids,
            'embedding_model': model_name,
            'index_stats': {
                'embedding_time': embed_time,
                'lexical_time': lex_time
            }
        }
        with open(self.get_index_path(version, 'metadata'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Indexes built successfully for version '{version}'")

    def load_all(self, version: str = 'default'):
        """Load all indexes for a given version."""
        if not self.is_indexed(version):
            raise FileNotFoundError(f"No indexes found for version '{version}'. Run indexing first.")
        
        model_name = self._get_config_val('embedding_model', 'jinaai/jina-embeddings-v2-base-code')
        device = self._get_config_val('device', 'cpu')
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        cache_dir = self._get_config_val('cache_dir', 'data/cache')
        
        logger.info(f"Loading indexes for version '{version}'...")
        
        # Load embedding index
        self.embedding_index = EmbeddingIndex(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir
        )
        self.embedding_index.load(self.get_index_path(version, 'faiss'))
        
        # Load lexical index
        self.lexical_index = LexicalIndex()
        self.lexical_index.load(self.get_index_path(version, 'lexical'))
        
        # Load metadata
        with open(self.get_index_path(version, 'metadata'), 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Indexes loaded: {self.metadata['num_documents']} documents")
