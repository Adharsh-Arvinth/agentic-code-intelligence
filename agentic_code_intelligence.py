"""
Agentic Code Intelligence - Python SDK Entry Point.

Provides a developer-friendly SDK for semantic code search, multi-version
code intelligence, and execution-guided reranking.
"""

from src.sdk.client import CodeIntelligenceClient, SearchResult

__version__ = "1.0.0"
__author__ = "Adharsh Arvinth, Rahul J, Sivamiruthula"
__all__ = ["CodeIntelligenceClient", "SearchResult"]
