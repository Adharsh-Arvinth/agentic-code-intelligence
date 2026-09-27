from typing import List, Dict
from src.indexing.embedding_index import EmbeddingIndex

class SemanticRetriever:
    def __init__(self, embedding_index: EmbeddingIndex, doc_ids: List[str] = None):
        self.embedding_index = embedding_index
        # Use doc_ids if passed, else wait to be used (not strictly specified in initial request if it was self contained, 
        # but required to map indices back to docs).
        self.doc_ids = doc_ids or []

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        query_embedding = self.embedding_index.encode_queries([query])[0]
        scores, indices = self.embedding_index.search(query_embedding, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx != -1 and idx < len(self.doc_ids):
                results.append({
                    'doc_id': self.doc_ids[idx],
                    'score': float(score),
                    'rank': rank + 1
                })
        return results

    def batch_retrieve(self, queries: List[str], top_k: int) -> Dict[str, List[Dict]]:
        query_embeddings = self.embedding_index.encode_queries(queries)
        scores_batch, indices_batch = self.embedding_index.batch_search(query_embeddings, top_k)
        
        batch_results = {}
        for q, scores, indices in zip(queries, scores_batch, indices_batch):
            results = []
            for rank, (score, idx) in enumerate(zip(scores, indices)):
                if idx != -1 and idx < len(self.doc_ids):
                    results.append({
                        'doc_id': self.doc_ids[idx],
                        'score': float(score),
                        'rank': rank + 1
                    })
            batch_results[q] = results
        return batch_results

    def normalize_scores(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return results
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        normalized = []
        for r in results:
            norm_score = 0.0 if max_score == min_score else (r['score'] - min_score) / (max_score - min_score)
            normalized.append({
                'doc_id': r['doc_id'],
                'score': norm_score,
                'rank': r['rank']
            })
        return normalized
