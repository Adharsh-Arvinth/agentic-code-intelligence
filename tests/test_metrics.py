"""Tests for evaluation metrics."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.metrics import ndcg_at_k, mrr, recall_at_k, map_at_k, evaluate_retrieval


def test_ndcg_perfect_ranking():
    """NDCG should be 1.0 for a perfect ranking."""
    relevance = [1, 1, 1, 0, 0]
    score = ndcg_at_k(relevance, k=3)
    assert score == pytest.approx(1.0)


def test_ndcg_worst_ranking():
    """NDCG should be less than 1 for imperfect ranking."""
    relevance = [0, 0, 1]
    score = ndcg_at_k(relevance, k=3)
    assert score < 1.0
    assert score > 0.0


def test_ndcg_empty():
    """NDCG should be 0 for empty input."""
    score = ndcg_at_k([], k=10)
    assert score == 0.0


def test_ndcg_no_relevant():
    """NDCG should be 0 when no relevant documents exist."""
    relevance = [0, 0, 0, 0]
    score = ndcg_at_k(relevance, k=4)
    assert score == 0.0


def test_mrr_first_position():
    """MRR should be 1.0 when first result is relevant."""
    assert mrr([1, 0, 0]) == 1.0


def test_mrr_second_position():
    """MRR should be 0.5 when second result is relevant."""
    assert mrr([0, 1, 0]) == pytest.approx(0.5)


def test_mrr_no_relevant():
    """MRR should be 0 when no relevant documents found."""
    assert mrr([0, 0, 0]) == 0.0


def test_recall_at_k():
    """Test recall@k computation."""
    retrieved = ['d1', 'd2', 'd3', 'd4', 'd5']
    relevant = {'d2', 'd4', 'd6'}
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2/3)


def test_recall_at_k_all_retrieved():
    """Test recall when all relevant docs are retrieved."""
    retrieved = ['d1', 'd2', 'd3']
    relevant = {'d1', 'd2'}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_map_at_k():
    """Test MAP@k computation."""
    relevance = [1, 0, 1, 0, 0]
    score = map_at_k(relevance, k=5)
    assert score > 0.0
    assert score <= 1.0


def test_evaluate_retrieval():
    """Test full evaluation pipeline."""
    qrels = {
        'q1': {'d1': 1, 'd3': 1},
        'q2': {'d2': 1}
    }
    results = {
        'q1': ['d1', 'd2', 'd3'],
        'q2': ['d1', 'd2', 'd3']
    }
    metrics = evaluate_retrieval(qrels, results, k_values=[1, 3])
    assert 'ndcg@1' in metrics
    assert 'ndcg@3' in metrics
    assert 'mrr' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())
