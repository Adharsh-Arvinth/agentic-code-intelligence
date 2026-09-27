from typing import List, Dict
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.lexical_retriever import LexicalRetriever

class HybridRetriever:
    def __init__(self, semantic_retriever: SemanticRetriever, lexical_retriever: LexicalRetriever, config: Dict):
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.config = config

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        fusion_method = self.config.get('fusion_method', 'rrf')
        sem_top_k = self.config.get('semantic_top_k', top_k * 2)
        lex_top_k = self.config.get('lexical_top_k', top_k * 2)

        semantic_results = self.semantic_retriever.retrieve(query, sem_top_k)
        lexical_results = self.lexical_retriever.retrieve(query, lex_top_k)

        if fusion_method == 'rrf':
            fused = self.rrf_fusion(semantic_results, lexical_results)
        elif fusion_method == 'weighted':
            fused = self.weighted_fusion(semantic_results, lexical_results, 
                                         self.config.get('semantic_weight', 0.7),
                                         self.config.get('lexical_weight', 0.3))
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        fused = sorted(fused, key=lambda x: x['score'], reverse=True)[:top_k]
        for rank, res in enumerate(fused):
            res['rank'] = rank + 1
            
        return fused

    def rrf_fusion(self, semantic_results: List[Dict], lexical_results: List[Dict], k: int = 60) -> List[Dict]:
        scores = {}
        
        for res in semantic_results:
            doc_id = res['doc_id']
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + res['rank'])
            
        for res in lexical_results:
            doc_id = res['doc_id']
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + res['rank'])
            
        return [{'doc_id': doc_id, 'score': score} for doc_id, score in scores.items()]

    def weighted_fusion(self, semantic_results: List[Dict], lexical_results: List[Dict], 
                        semantic_weight: float = 0.7, lexical_weight: float = 0.3) -> List[Dict]:
        
        sem_norm = self.semantic_retriever.normalize_scores(semantic_results)
        lex_norm = self.lexical_retriever.normalize_scores(lexical_results)
        
        scores = {}
        for res in sem_norm:
            scores[res['doc_id']] = scores.get(res['doc_id'], 0.0) + res['score'] * semantic_weight
            
        for res in lex_norm:
            scores[res['doc_id']] = scores.get(res['doc_id'], 0.0) + res['score'] * lexical_weight
            
        return [{'doc_id': doc_id, 'score': score} for doc_id, score in scores.items()]

    def batch_retrieve(self, queries: List[str], top_k: int) -> Dict[str, List[Dict]]:
        return {q: self.retrieve(q, top_k) for q in queries}
