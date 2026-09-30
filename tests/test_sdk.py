"""Unit tests for the Agentic Code Intelligence Python SDK."""

import pytest
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic_code_intelligence import CodeIntelligenceClient, SearchResult


def test_sdk_imports():
    """Verify SDK classes are properly exported."""
    assert CodeIntelligenceClient is not None
    assert SearchResult is not None


def test_sdk_search_result_dataclass():
    """Test SearchResult dataclass serialization."""
    res = SearchResult(
        rank=1,
        doc_id="d100",
        score=0.95,
        code="print('hello')",
        version="v1.0",
        metadata={"author": "team"}
    )
    d = res.to_dict()
    assert d["rank"] == 1
    assert d["doc_id"] == "d100"
    assert d["score"] == 0.95
    assert d["code"] == "print('hello')"
    assert d["version"] == "v1.0"
    assert d["metadata"]["author"] == "team"


def test_sdk_client_initialization():
    """Test initializing the SDK client with CPU defaults."""
    client = CodeIntelligenceClient(device="cpu")
    status = client.health_check()
    assert isinstance(status, dict)
    assert status["device"] == "cpu"
    assert "available_versions" in status
    assert "embedding_model" in status


def test_sdk_verify_execution_matched():
    """Test AST-guarded sandboxed execution verification via SDK."""
    client = CodeIntelligenceClient(device="cpu")
    sample_input = "5\n"
    expected_output = "25"
    code = "n = int(input())\nprint(n * n)\n"

    result = client.verify_execution(
        sample_input=sample_input,
        expected_output=expected_output,
        code_snippet=code
    )
    assert result["matched"] is True
    assert result["boost"] > 0.0


def test_sdk_verify_execution_mismatched():
    """Test execution verification with mismatched output."""
    client = CodeIntelligenceClient(device="cpu")
    sample_input = "5\n"
    expected_output = "999"
    code = "n = int(input())\nprint(n * n)\n"

    result = client.verify_execution(
        sample_input=sample_input,
        expected_output=expected_output,
        code_snippet=code
    )
    assert result["matched"] is False
    assert result["boost"] == 0.0


def test_sdk_list_versions():
    """Test listing available repository versions via SDK."""
    client = CodeIntelligenceClient(device="cpu")
    versions = client.list_versions()
    assert isinstance(versions, list)
