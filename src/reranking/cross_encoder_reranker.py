import torch
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', device: str = 'cpu'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        
        # CrossEncoder expects pairs of (query, document)
        pairs = [[query, doc.get('text', '')[:2000]] for doc in candidates]  # truncated to avoid too long
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Update scores and sort
        for i, doc in enumerate(candidates):
            doc['score'] = float(scores[i])
            
        reranked = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return reranked[:top_k]

    def batch_rerank(self, queries_and_candidates: dict, top_k: int) -> dict:
        """
        queries_and_candidates: dict mapping query_id to {'query': str, 'candidates': list[dict]}
        """
        result = {}
        for q_id, data in queries_and_candidates.items():
            result[q_id] = self.rerank(data['query'], data['candidates'], top_k)
        return result
