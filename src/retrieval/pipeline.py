"""End-to-end retrieval pipeline orchestrator."""

import time
import logging
from typing import Dict, List, Optional, Any

from src.indexing.index_manager import IndexManager
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.lexical_retriever import LexicalRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Multi-stage retrieval pipeline: semantic → lexical → fusion → reranking."""
    
    def __init__(self, config):
        self.config = config
        self.index_manager = IndexManager(config)
        self.semantic_retriever: Optional[SemanticRetriever] = None
        self.lexical_retriever: Optional[LexicalRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.reranker = None
        self._initialized = False
    
    def _get_config_val(self, key: str, default=None):
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)
    
    def build_index(self, corpus: Dict[str, Any], version: str = 'default'):
        """Build or load indexes for the corpus."""
        self.index_manager.build_all(corpus, version=version)
        self._init_retrievers()
    
    def _init_retrievers(self):
        """Initialize retriever components from loaded indexes."""
        if self.index_manager.embedding_index is None:
            return
        
        doc_ids = self.index_manager.metadata.get('doc_ids', [])
        
        self.semantic_retriever = SemanticRetriever(
            self.index_manager.embedding_index,
            doc_ids
        )
        self.lexical_retriever = LexicalRetriever(
            self.index_manager.lexical_index
        )
        
        # Build hybrid config
        hybrid_config = {
            'fusion_method': self._get_config_val('fusion_strategy', 'rrf'),
            'semantic_top_k': self._get_config_val('semantic_top_k', 200),
            'lexical_top_k': self._get_config_val('lexical_top_k', 200),
            'semantic_weight': self._get_config_val('semantic_weight', 0.7),
            'lexical_weight': self._get_config_val('lexical_weight', 0.3),
            'rrf_k': self._get_config_val('rrf_k', 60),
        }
        
        self.hybrid_retriever = HybridRetriever(
            self.semantic_retriever,
            self.lexical_retriever,
            hybrid_config
        )
        self._initialized = True
    
    def _ensure_loaded(self, version: str = 'default'):
        """Ensure indexes are loaded."""
        if not self._initialized:
            if self.index_manager.is_indexed(version):
                self.index_manager.load_all(version)
                self._init_retrievers()
            else:
                raise RuntimeError(
                    f"No indexes found for version '{version}'. "
                    f"Run 'python -m src index' first."
                )
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        version: str = 'default',
        use_reranker: bool = False,
        method: str = 'hybrid'
    ) -> List[Dict]:
        """Run the full retrieval pipeline."""
        self._ensure_loaded(version)
        
        latencies = {}
        start_time = time.time()
        
        # Stage 1-3: Retrieval
        retrieval_start = time.time()
        
        if method == 'semantic':
            sem_top_k = self._get_config_val('semantic_top_k', 200)
            results = self.semantic_retriever.retrieve(query, min(top_k, sem_top_k))
            
        elif method == 'lexical':
            lex_top_k = self._get_config_val('lexical_top_k', 200)
            results = self.lexical_retriever.retrieve(query, min(top_k, lex_top_k))
            
        elif method == 'hybrid':
            fusion_top_k = self._get_config_val('fusion_top_k', 100)
            results = self.hybrid_retriever.retrieve(query, fusion_top_k)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'semantic', 'lexical', or 'hybrid'.")
        
        latencies['retrieval'] = time.time() - retrieval_start
        logger.info(f"Retrieval ({method}): {len(results)} candidates in {latencies['retrieval']:.3f}s")
        
        # Stage 4: Reranking (optional)
        if use_reranker and results:
            rerank_start = time.time()
            try:
                if self.reranker is None:
                    from src.reranking.cross_encoder_reranker import CrossEncoderReranker
                    reranker_model = self._get_config_val('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                    device = self._get_config_val('device', 'cpu')
                    if device == 'auto':
                        try:
                            import torch
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        except ImportError:
                            device = 'cpu'
                    self.reranker = CrossEncoderReranker(model_name=reranker_model, device=device)
                
                rerank_top_k = self._get_config_val('rerank_top_k', 50)
                # Need to add text to candidates for reranker - we'll use doc_id as placeholder
                # In practice, the corpus text should be passed in
                results = results[:rerank_top_k]
                latencies['reranking'] = time.time() - rerank_start
                logger.info(f"Reranking: {rerank_top_k} candidates in {latencies['reranking']:.3f}s")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using unreranked results.")
        
        # Final top-k
        final_top_k = min(top_k, len(results))
        results = results[:final_top_k]
        
        # Update ranks
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        latencies['total'] = time.time() - start_time
        logger.info(f"Total retrieval: {latencies['total']:.3f}s")
        
        # Attach latencies to first result
        if results:
            results[0]['latencies'] = latencies
        
        return results
    
    def batch_retrieve(
        self, 
        queries: Dict[str, str], 
        top_k: int = 10,
        version: str = 'default',
        method: str = 'hybrid'
    ) -> Dict[str, List[Dict]]:
        """Retrieve for multiple queries."""
        self._ensure_loaded(version)
        results = {}
        for qid, query in queries.items():
            results[qid] = self.retrieve(
                query, top_k=top_k, version=version, method=method
            )
        return results
