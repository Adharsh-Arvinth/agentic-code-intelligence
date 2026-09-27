"""Tests for the pipeline and MTEB wrapper."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config, load_config


def test_config_defaults():
    config = Config()
    assert config.embedding_model == 'jinaai/jina-embeddings-v2-base-code'
    assert config.batch_size == 64
    assert config.fusion_strategy == 'rrf'


def test_config_from_yaml():
    config = load_config('configs/default.yaml')
    assert isinstance(config, Config)
    assert config.embedding_model is not None


def test_config_device_detection():
    config = Config(device='cpu')
    assert config.get_device() == 'cpu'


def test_config_auto_device():
    config = Config(device='auto')
    device = config.get_device()
    assert device in ['cpu', 'cuda']


def test_mteb_wrapper_interface():
    """Verify CodeRetrievalModel has required MTEB interface methods."""
    from src.evaluation.mteb_wrapper import CodeRetrievalModel
    # Check that the class has the required methods
    assert hasattr(CodeRetrievalModel, 'encode')
    assert hasattr(CodeRetrievalModel, 'encode_queries')
    assert hasattr(CodeRetrievalModel, 'encode_corpus')


def test_empty_query_handling():
    """Test that empty query doesn't crash preprocessing."""
    from src.preprocessing.query_preprocessor import QueryPreprocessor
    qp = QueryPreprocessor()
    result = qp.process("")
    assert result['original'] == ""
