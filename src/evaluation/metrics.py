import math

def ndcg_at_k(relevance_scores: list[int], k: int) -> float:
    scores = relevance_scores[:k]
    dcg = sum((2**score - 1) / math.log2(idx + 2) for idx, score in enumerate(scores))
    
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2**score - 1) / math.log2(idx + 2) for idx, score in enumerate(ideal_scores))
    
    return dcg / idcg if idcg > 0 else 0.0

def mrr(relevance_scores: list[int]) -> float:
    for idx, score in enumerate(relevance_scores):
        if score > 0:
            return 1.0 / (idx + 1)
    return 0.0

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_retrieved / len(relevant)

def map_at_k(relevance_scores: list[int], k: int) -> float:
    scores = relevance_scores[:k]
    relevant_count = 0
    precision_sum = 0.0
    
    for idx, score in enumerate(scores):
        if score > 0:
            relevant_count += 1
            precision_sum += relevant_count / (idx + 1)
            
    total_relevant = sum(1 for s in relevance_scores if s > 0)
    if total_relevant == 0:
        return 0.0
    return precision_sum / total_relevant

def evaluate_retrieval(qrels: dict, results: dict, k_values: list[int]) -> dict:
    metrics = {f'ndcg@{k}': [] for k in k_values}
    metrics.update({f'recall@{k}': [] for k in k_values})
    metrics.update({f'map@{k}': [] for k in k_values})
    metrics['mrr'] = []
    
    for q_id, relevant_docs in qrels.items():
        if q_id not in results:
            continue
            
        retrieved_docs = results[q_id]
        
        relevance_scores = [relevant_docs.get(doc, 0) for doc in retrieved_docs]
        relevant_set = {doc for doc, rel in relevant_docs.items() if rel > 0}
        
        metrics['mrr'].append(mrr(relevance_scores))
        
        for k in k_values:
            metrics[f'ndcg@{k}'].append(ndcg_at_k(relevance_scores, k))
            metrics[f'recall@{k}'].append(recall_at_k(retrieved_docs, relevant_set, k))
            metrics[f'map@{k}'].append(map_at_k(relevance_scores, k))
            
    avg_metrics = {m: sum(vals)/len(vals) if vals else 0.0 for m, vals in metrics.items()}
    return avg_metrics
