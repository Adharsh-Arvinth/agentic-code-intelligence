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
