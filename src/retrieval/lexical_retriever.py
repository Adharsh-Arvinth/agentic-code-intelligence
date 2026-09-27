from typing import List, Dict
from src.indexing.lexical_index import LexicalIndex

class LexicalRetriever:
    def __init__(self, lexical_index: LexicalIndex):
        self.lexical_index = lexical_index

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        raw_results = self.lexical_index.search(query, top_k)
        results = []
        for rank, (doc_id, score) in enumerate(raw_results):
            results.append({
                'doc_id': doc_id,
                'score': score,
                'rank': rank + 1
            })
        return results

    def batch_retrieve(self, queries: List[str], top_k: int) -> Dict[str, List[Dict]]:
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, top_k)
        return results

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
