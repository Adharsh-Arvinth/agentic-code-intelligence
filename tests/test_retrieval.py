"""Tests for retrieval components."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval.hybrid_retriever import HybridRetriever


def test_rrf_fusion():
    """Test Reciprocal Rank Fusion with known inputs."""
    # Create mock retrievers
    class MockRetriever:
        def __init__(self, results):
            self._results = results
        def retrieve(self, query, top_k):
            return self._results[:top_k]
        def normalize_scores(self, results):
            return results
    
    sem_results = [
        {'doc_id': 'doc1', 'score': 0.9, 'rank': 1},
        {'doc_id': 'doc2', 'score': 0.7, 'rank': 2},
        {'doc_id': 'doc3', 'score': 0.5, 'rank': 3},
    ]
    lex_results = [
        {'doc_id': 'doc3', 'score': 0.8, 'rank': 1},
        {'doc_id': 'doc1', 'score': 0.6, 'rank': 2},
        {'doc_id': 'doc4', 'score': 0.4, 'rank': 3},
    ]
    
    config = {'fusion_method': 'rrf', 'semantic_top_k': 10, 'lexical_top_k': 10}
    hybrid = HybridRetriever(
        MockRetriever(sem_results),
        MockRetriever(lex_results),
        config
    )
    
    results = hybrid.retrieve("test query", top_k=4)
    assert len(results) == 4
    doc_ids = [r['doc_id'] for r in results]
    # doc1 and doc3 appear in both lists, so they should rank higher
    assert 'doc1' in doc_ids[:2] or 'doc3' in doc_ids[:2]


def test_weighted_fusion():
    """Test weighted score fusion."""
    class MockRetriever:
        def __init__(self, results):
            self._results = results
        def retrieve(self, query, top_k):
            return self._results[:top_k]
        def normalize_scores(self, results):
            return results
    
    sem_results = [
        {'doc_id': 'd1', 'score': 0.8, 'rank': 1},
        {'doc_id': 'd2', 'score': 0.4, 'rank': 2},
    ]
    lex_results = [
        {'doc_id': 'd1', 'score': 0.2, 'rank': 1},
        {'doc_id': 'd2', 'score': 0.6, 'rank': 2},
    ]
    
    config = {
        'fusion_method': 'weighted',
        'semantic_top_k': 10,
        'lexical_top_k': 10,
        'semantic_weight': 0.5,
        'lexical_weight': 0.5
    }
    hybrid = HybridRetriever(
        MockRetriever(sem_results),
        MockRetriever(lex_results),
        config
    )
    
    results = hybrid.retrieve("test", top_k=2)
    assert len(results) == 2


def test_empty_results():
    """Test with empty result sets."""
    class MockRetriever:
        def retrieve(self, query, top_k):
            return []
        def normalize_scores(self, results):
            return results
    
    config = {'fusion_method': 'rrf', 'semantic_top_k': 10, 'lexical_top_k': 10}
    hybrid = HybridRetriever(MockRetriever(), MockRetriever(), config)
    results = hybrid.retrieve("empty", top_k=5)
    assert len(results) == 0


def test_single_result():
    """Test with only one result."""
    class MockRetriever:
        def __init__(self, results=None):
            self._results = results or []
        def retrieve(self, query, top_k):
            return self._results[:top_k]
        def normalize_scores(self, results):
            return results
    
    sem_results = [{'doc_id': 'd1', 'score': 0.9, 'rank': 1}]
    config = {'fusion_method': 'rrf', 'semantic_top_k': 10, 'lexical_top_k': 10}
    hybrid = HybridRetriever(
        MockRetriever(sem_results),
        MockRetriever([]),
        config
    )
    results = hybrid.retrieve("test", top_k=5)
    assert len(results) == 1
    assert results[0]['doc_id'] == 'd1'
