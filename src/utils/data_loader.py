"""Load CoIR AppsRetrieval dataset from HuggingFace."""

import os
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def load_dataset_splits(cache_dir: str = 'data/cache') -> Dict[str, Any]:
    """
    Load CoIR AppsRetrieval dataset splits from HuggingFace.
    
    Returns:
        Dict with 'train' and 'test' keys, each containing:
            'queries': {query_id: query_text}
            'corpus': {doc_id: {'title': str, 'text': str}}
            'qrels': {query_id: {doc_id: relevance_score}}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    logger.info("Loading CoIR AppsRetrieval dataset from HuggingFace...")
    
    try:
        corpus_ds = load_dataset('CoIR-Retrieval/apps', 'corpus', cache_dir=cache_dir)
        queries_ds = load_dataset('CoIR-Retrieval/apps', 'queries', cache_dir=cache_dir)
        qrels_ds = load_dataset('CoIR-Retrieval/apps', 'default', cache_dir=cache_dir)
    except Exception as e:
        logger.warning(f"Primary CoIR-Retrieval/apps load failed ({e}), trying mteb/AppsRetrieval...")
        try:
            corpus_ds = load_dataset('mteb/AppsRetrieval', 'corpus', cache_dir=cache_dir)
            queries_ds = load_dataset('mteb/AppsRetrieval', 'queries', cache_dir=cache_dir)
            qrels_ds = load_dataset('mteb/AppsRetrieval', 'default', cache_dir=cache_dir)
        except Exception as e2:
            raise RuntimeError(
                f"Could not load AppsRetrieval dataset. "
                f"First error: {e}. Second error: {e2}."
            )
    
    # Parse corpus (shared across splits)
    corpus_dict = {}
    for split_name in corpus_ds:
        for item in corpus_ds[split_name]:
            doc_id = str(item.get('_id', item.get('id', '')))
            corpus_dict[doc_id] = {
                'title': str(item.get('title', '')),
                'text': str(item.get('text', ''))
            }
    
    # Parse all queries into a master dictionary
    all_queries = {}
    for split_name in queries_ds:
        for item in queries_ds[split_name]:
            qid = str(item.get('_id', item.get('id', '')))
            all_queries[qid] = str(item.get('text', ''))
    
    # Parse qrels per split ('train' has 5000, 'test' has 3765) and map corresponding queries
    result = {}
    if qrels_ds is not None:
        for split_name in qrels_ds:
            qrels = {}
            split_queries = {}
            for item in qrels_ds[split_name]:
                qid = str(item.get('query-id', item.get('query_id', '')))
                docid = str(item.get('corpus-id', item.get('corpus_id', '')))
                score = int(item.get('score', 1))
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = score
                if qid in all_queries:
                    split_queries[qid] = all_queries[qid]
            result[split_name] = {
                'queries': split_queries,
                'corpus': corpus_dict,
                'qrels': qrels
            }
    
    # Ensure both train and test exist
    for split in ['train', 'test']:
        if split not in result:
            result[split] = {'queries': {}, 'corpus': corpus_dict, 'qrels': {}}
    
    # Print statistics
    logger.info(f"Corpus: {len(corpus_dict)} documents")
    for split_name, split_data in result.items():
        n_queries = len(split_data['queries'])
        n_qrels = len(split_data['qrels'])
        logger.info(f"Split '{split_name}': {n_queries} queries, {n_qrels} qrels")
    
    return result


def load_test_split(cache_dir: str = 'data/cache') -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Dict[str, int]]]:
    """Convenience function to load only the test split."""
    data = load_dataset_splits(cache_dir=cache_dir)
    test_data = data.get('test', {})
    return (
        test_data.get('queries', {}),
        test_data.get('corpus', {}),
        test_data.get('qrels', {})
    )


def load_val_split(max_queries: int = 500, cache_dir: str = 'data/cache') -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Dict[str, int]]]:
    """Load validation subset from the train split (strictly disjoint from test split)."""
    data = load_dataset_splits(cache_dir=cache_dir)
    train_data = data.get('train', {})
    queries = train_data.get('queries', {})
    corpus = train_data.get('corpus', {})
    qrels = train_data.get('qrels', {})
    if max_queries and len(queries) > max_queries:
        qids = sorted(list(queries.keys()))[:max_queries]
        queries = {q: queries[q] for q in qids}
        qrels = {q: qrels[q] for q in qids if q in qrels}
    return queries, corpus, qrels
