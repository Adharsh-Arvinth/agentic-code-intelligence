"""Structured logging and performance tracking utilities."""

import logging
import time
import sys
from typing import Dict, List, Any, Optional
from contextlib import contextmanager


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


_module_logger = setup_logger(__name__)


@contextmanager
def Timer(description: str):
    """Context manager to time a block of code."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _module_logger.info(f"{description} took {elapsed:.4f} seconds")


class PerformanceTracker:
    """Track performance metrics across pipeline stages."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'indexing_time': [],
            'query_latency': [],
            'embedding_time': [],
            'reranking_time': []
        }
        self._starts: Dict[str, float] = {}

    def start(self, metric_name: str) -> None:
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self._starts[metric_name] = time.perf_counter()

    def stop(self, metric_name: str) -> float:
        if metric_name in self._starts:
            elapsed = time.perf_counter() - self._starts[metric_name]
            self.metrics[metric_name].append(elapsed)
            del self._starts[metric_name]
            return elapsed
        return 0.0
        
    def get_average(self, metric_name: str) -> float:
        times = self.metrics.get(metric_name, [])
        if not times:
            return 0.0
        return sum(times) / len(times)
        
    def print_summary(self) -> None:
        _module_logger.info("--- Performance Summary ---")
        for metric, times in self.metrics.items():
            if times:
                avg = sum(times) / len(times)
                _module_logger.info(f"{metric}: Avg = {avg:.4f}s, Total = {sum(times):.4f}s, Count = {len(times)}")


def format_results(
    results: List[Dict[str, Any]], 
    corpus: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """Pretty-print retrieval results to console."""
    log = logger or _module_logger
    
    if not results:
        log.info("No results found.")
        return
    
    log.info(f"--- Retrieved {len(results)} results ---")
    for idx, res in enumerate(results):
        doc_id = res.get('doc_id', res.get('docid', 'N/A'))
        score = res.get('score', 0.0)
        rank = res.get('rank', idx + 1)
        
        log.info(f"")
        log.info(f"  Rank {rank} | Doc: {doc_id} | Score: {score:.6f}")
        
        # Show code snippet if corpus is available
        if corpus and doc_id in corpus:
            doc = corpus[doc_id]
            if isinstance(doc, dict):
                code = doc.get('text', '')
            else:
                code = str(doc)
            # Show first 5 lines of code
            lines = code.split('\n')[:5]
            preview = '\n    '.join(lines)
            if len(code.split('\n')) > 5:
                preview += '\n    ...'
            log.info(f"  Code preview:")
            log.info(f"    {preview}")
        
        log.info(f"  {'─' * 50}")
