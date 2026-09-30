"""Developer SDK Client for Agentic Code Intelligence."""

import os
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

from src.utils.config import Config, load_config
from src.retrieval.pipeline import RetrievalPipeline
from src.reranking.execution_verifier import ExecutionVerifier
from src.versioning.version_manager import VersionManager

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured result returned by CodeIntelligenceClient.search()."""
    rank: int
    doc_id: str
    score: float
    code: str
    version: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to standard dictionary."""
        return asdict(self)


class CodeIntelligenceClient:
    """
    High-level Python SDK Client for Agentic Code Intelligence.

    Provides a clean, developer-friendly interface for:
      - Natural-language code retrieval across multi-version repositories
      - AST-guarded sandboxed execution verification
      - Index building, inspection, and lifecycle management

    Example:
        >>> from agentic_code_intelligence import CodeIntelligenceClient
        >>> client = CodeIntelligenceClient()
        >>> results = client.search("Find shortest path using Dijkstra algorithm", top_k=3)
        >>> for r in results:
        ...     print(f"Rank {r.rank} | Doc {r.doc_id} | Score {r.score:.4f}")
        ...     print(r.code)
    """

    def __init__(
        self,
        config_path: Optional[str] = "configs/default.yaml",
        model_name: Optional[str] = None,
        device: str = "cpu",
        cache_dir: str = "data/cache",
        index_dir: str = "data/indexes",
    ):
        """Initialize the CodeIntelligenceClient with configuration defaults."""
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            self.config = Config()

        if model_name:
            self.config.embedding_model = model_name
        if device:
            self.config.device = device
        if cache_dir:
            self.config.cache_dir = cache_dir
        if index_dir:
            self.config.index_dir = index_dir

        self._pipeline: Optional[RetrievalPipeline] = None
        self._verifier: Optional[ExecutionVerifier] = None
        self._version_manager = VersionManager(base_dir=self.config.index_dir)

    @property
    def pipeline(self) -> RetrievalPipeline:
        """Lazily initialize and return the underlying RetrievalPipeline."""
        if self._pipeline is None:
            self._pipeline = RetrievalPipeline(self.config)
        return self._pipeline

    @property
    def verifier(self) -> ExecutionVerifier:
        """Lazily initialize and return the AST-guarded ExecutionVerifier."""
        if self._verifier is None:
            self._verifier = ExecutionVerifier()
        return self._verifier

    def search(
        self,
        query: str,
        top_k: int = 10,
        version: str = "default",
        use_reranker: bool = False,
        method: str = "hybrid",
    ) -> List[SearchResult]:
        """
        Search code snippets using natural language.

        Args:
            query: Natural-language algorithmic problem or developer intent query.
            top_k: Number of ranked code results to return.
            version: Code version tag to query against (default: 'default').
            use_reranker: Whether to apply neural Cross-Encoder reranking.
            method: Retrieval method ('hybrid', 'semantic', or 'lexical').

        Returns:
            List of SearchResult objects with rank, doc_id, score, code, and metadata.
        """
        raw_results = self.pipeline.retrieve(
            query=query,
            top_k=top_k,
            version=version,
            use_reranker=use_reranker,
            method=method,
        )

        results = []
        for r in raw_results:
            results.append(
                SearchResult(
                    rank=r.get("rank", 0),
                    doc_id=r.get("doc_id", "N/A"),
                    score=float(r.get("score", 0.0)),
                    code=r.get("code", ""),
                    version=r.get("version", version),
                    metadata=r.get("metadata", {}),
                )
            )
        return results

    def verify_execution(
        self,
        sample_input: str,
        expected_output: str,
        code_snippet: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run AST-guarded sandboxed execution verification against sample I/O.

        Args:
            sample_input: String input fed to stdin.
            expected_output: Expected output string from stdout.
            code_snippet: Python code to compile and test on-the-fly.
            doc_id: Existing corpus document ID (e.g. 'd4626') to verify.

        Returns:
            Dict containing match status, boost score, and verified doc_id.
        """
        target_id = doc_id
        if code_snippet:
            target_id = "__sdk_eval__"
            self.verifier.compile_corpus({target_id: code_snippet})
        elif target_id is None:
            raise ValueError("Must provide either 'code_snippet' or 'doc_id'")

        boost = self.verifier.verify_single(target_id, sample_input, expected_output)
        return {
            "doc_id": target_id,
            "matched": boost > 0.0,
            "boost": boost,
        }

    def list_versions(self) -> List[str]:
        """List all indexed code versions available in the repository."""
        return self._version_manager.list_versions()

    def get_version_metadata(self, version: str) -> Dict[str, Any]:
        """Retrieve stored metadata for a registered version tag."""
        return self._version_manager.get_version_metadata(version)

    def health_check(self) -> Dict[str, Any]:
        """Check system and index readiness status."""
        indexed_versions = self.list_versions()
        return {
            "status": "ready" if indexed_versions else "not_indexed",
            "device": self.config.get_device(),
            "embedding_model": self.config.embedding_model,
            "available_versions": indexed_versions,
            "version_count": len(indexed_versions),
        }
