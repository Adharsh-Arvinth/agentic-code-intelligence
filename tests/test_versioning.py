"""Tests for version management."""

import pytest
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.versioning.version_manager import VersionManager


def test_register_version():
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VersionManager(base_dir=tmpdir)
        vm.register_version('v1', {'repo': 'test', 'commit': 'abc123'})
        assert 'v1' in vm.list_versions()


def test_list_versions_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VersionManager(base_dir=tmpdir)
        versions = vm.list_versions()
        assert isinstance(versions, list)


def test_get_version_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VersionManager(base_dir=tmpdir)
        metadata = {'repo': 'test', 'commit': 'abc123', 'language': 'python'}
        vm.register_version('v1', metadata)
        result = vm.get_version_metadata('v1')
        assert result['repo'] == 'test'
        assert result['commit'] == 'abc123'


def test_filter_by_version():
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VersionManager(base_dir=tmpdir)
        results = [
            {'doc_id': 'd1', 'score': 0.9, 'version': 'v1'},
            {'doc_id': 'd2', 'score': 0.8, 'version': 'v2'},
            {'doc_id': 'd3', 'score': 0.7, 'version': 'v1'},
        ]
        filtered = vm.filter_results_by_version(results, 'v1')
        assert len(filtered) == 2
        assert all(r['version'] == 'v1' for r in filtered)


def test_metadata_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VersionManager(base_dir=tmpdir)
        vm.register_version('v1', {'repo': 'test'})
        
        # Save and reload
        save_path = os.path.join(tmpdir, 'versions.json')
        vm.save_metadata(save_path)
        
        vm2 = VersionManager(base_dir=tmpdir)
        vm2.load_metadata(save_path)
        assert 'v1' in vm2.list_versions()


def test_deduplication_and_priority():
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VersionManager(base_dir=tmpdir)
        results = [
            {'doc_id': 'd1', 'score': 0.85, 'version': 'v1.0'},
            {'doc_id': 'd1', 'score': 0.92, 'version': 'v2.0'},
            {'doc_id': 'd2', 'score': 0.75, 'version': 'v1.0'},
        ]
        dedup_latest = vm.deduplicate_results(results, version_priority='latest')
        assert len(dedup_latest) == 2
        d1_entry = [r for r in dedup_latest if r['doc_id'] == 'd1'][0]
        assert d1_entry['version'] == 'v2.0'


def test_end_to_end_version_retrieval():
    """Synthetic version test validating multi-version registration, indexing, rebuilding, and isolated retrieval."""
    from src.utils.config import Config
    from src.retrieval.pipeline import RetrievalPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(embedding_model='BAAI/bge-small-en-v1.5', index_dir=tmpdir, device='cpu', batch_size=8)
        pipeline = RetrievalPipeline(cfg)

        corpus_v1 = {
            'v1_dijkstra': {'title': 'Dijkstra v1', 'text': 'import heapq\ndef shortest_path_v1(graph, src):\n    pq = [(0, src)]\n    return pq\n'},
            'v1_sort': {'title': 'Merge Sort v1', 'text': 'def merge_sort_v1(arr):\n    return sorted(arr)\n'},
        }
        corpus_v2 = {
            'v2_dijkstra': {'title': 'Dijkstra v2', 'text': 'import heapq\ndef shortest_path_v2_optimized(graph, start, target):\n    heap = [(0, start)]\n    return heap\n'},
            'v2_dp': {'title': 'Knapsack v2', 'text': 'def knapsack_dp_v2(weights, values, cap):\n    dp = [0] * (cap + 1)\n    return dp[-1]\n'},
        }

        pipeline.build_index(corpus_v1, version='v1.0', metadata={'commit': '111aaa'})
        pipeline.build_index(corpus_v2, version='v2.0', metadata={'commit': '222bbb'})

        assert 'v1.0' in pipeline.version_manager.list_versions()
        assert 'v2.0' in pipeline.version_manager.list_versions()
        assert pipeline.version_manager.get_version_metadata('v1.0')['commit'] == '111aaa'

        res_v1 = pipeline.retrieve("shortest path using Dijkstra heapq", top_k=2, version='v1.0', use_reranker=False)
        res_v2 = pipeline.retrieve("shortest path using Dijkstra heapq", top_k=2, version='v2.0', use_reranker=False)

        assert all(r['version'] == 'v1.0' for r in res_v1)
        assert all(r['version'] == 'v2.0' for r in res_v2)
        assert res_v1[0]['doc_id'].startswith('v1_')
        assert res_v2[0]['doc_id'].startswith('v2_')

        # Test force rebuild on v1.0
        pipeline.build_index(corpus_v1, version='v1.0', force=True)
        res_v1_rebuilt = pipeline.retrieve("merge sort array", top_k=1, version='v1.0', use_reranker=False)
        assert res_v1_rebuilt[0]['doc_id'] == 'v1_sort'

