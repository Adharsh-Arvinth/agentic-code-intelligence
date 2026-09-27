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
        # Try loading from mteb namespace first
        try:
            corpus_ds = load_dataset('mteb/AppsRetrieval', 'corpus', cache_dir=cache_dir, trust_remote_code=True)
            queries_ds = load_dataset('mteb/AppsRetrieval', 'queries', cache_dir=cache_dir, trust_remote_code=True)
            default_ds = load_dataset('mteb/AppsRetrieval', 'default', cache_dir=cache_dir, trust_remote_code=True)
        except Exception:
            # Fallback to CoIR namespace
            corpus_ds = load_dataset('CoIR-Retrieval/apps', 'corpus', cache_dir=cache_dir, trust_remote_code=True)
            queries_ds = load_dataset('CoIR-Retrieval/apps', 'queries', cache_dir=cache_dir, trust_remote_code=True)
            default_ds = load_dataset('CoIR-Retrieval/apps', 'default', cache_dir=cache_dir, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Attempting alternative loading method...")
        # Try loading as a single dataset
        try:
            ds = load_dataset('CoIR-Retrieval/apps', cache_dir=cache_dir, trust_remote_code=True)
            return _parse_single_dataset(ds)
        except Exception as e2:
            raise RuntimeError(
                f"Could not load AppsRetrieval dataset. "
                f"First error: {e}. Second error: {e2}. "
                f"Make sure you have internet access and the 'datasets' package installed."
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
    
    # Parse queries per split
    result = {}
    for split_name in queries_ds:
        queries = {}
        for item in queries_ds[split_name]:
            qid = str(item.get('_id', item.get('id', '')))
            queries[qid] = str(item.get('text', ''))
        
        result[split_name] = {
            'queries': queries,
            'corpus': corpus_dict,
            'qrels': {}
        }
    
    # Parse qrels
    if default_ds is not None:
        for split_name in default_ds:
            qrels = {}
            for item in default_ds[split_name]:
                qid = str(item.get('query-id', item.get('query_id', '')))
                docid = str(item.get('corpus-id', item.get('corpus_id', '')))
                score = int(item.get('score', 1))
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = score
            if split_name in result:
                result[split_name]['qrels'] = qrels
            else:
                result[split_name] = {
                    'queries': {},
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


def _parse_single_dataset(ds) -> Dict[str, Any]:
    """Parse a single dataset format (fallback)."""
    corpus_dict = {}
    result = {}
    
    for split_name in ds:
        queries = {}
        qrels = {}
        for item in ds[split_name]:
            # Try to extract query, corpus, and qrels from the item
            qid = str(item.get('query-id', item.get('_id', '')))
            query_text = str(item.get('query', item.get('text', '')))
            doc_id = str(item.get('corpus-id', ''))
            doc_text = str(item.get('corpus-text', item.get('positive', '')))
            
            if query_text:
                queries[qid] = query_text
            if doc_id and doc_text:
                corpus_dict[doc_id] = {'title': '', 'text': doc_text}
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = 1
        
        result[split_name] = {
            'queries': queries,
            'corpus': corpus_dict,
            'qrels': qrels
        }
    
    return result


def load_test_split(cache_dir: str = 'data/cache'):
    """Convenience function to load only the test split."""
    data = load_dataset_splits(cache_dir=cache_dir)
    test_data = data.get('test', {})
    return (
        test_data.get('queries', {}),
        test_data.get('corpus', {}),
        test_data.get('qrels', {})
    )
