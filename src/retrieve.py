"""Run retrieval queries against the indexed corpus."""

import argparse
import time
import json


def run_retrieve(args):
    """Run a retrieval query and display results."""
    from src.utils.config import load_config
    from src.utils.data_loader import load_dataset_splits
    from src.utils.logging_utils import setup_logger, Timer, format_results
    from src.retrieval.pipeline import RetrievalPipeline

    logger = setup_logger("retrieve")
    config = load_config(args.config if hasattr(args, 'config') else "configs/default.yaml")

    query = args.query
    top_k = args.top_k if hasattr(args, 'top_k') else 10
    version = args.version if hasattr(args, 'version') else "default"
    use_reranker = not (args.no_rerank if hasattr(args, 'no_rerank') else False)
    method = args.method if hasattr(args, 'method') else "hybrid"

    logger.info("=" * 60)
    logger.info("AGENTIC CODE INTELLIGENCE - RETRIEVAL")
    logger.info("=" * 60)
    logger.info(f"Query: {query}")
    logger.info(f"Method: {method}")
    logger.info(f"Top-K: {top_k}")
    logger.info(f"Version: {version}")
    logger.info(f"Reranking: {'enabled' if use_reranker else 'disabled'}")
    logger.info("-" * 60)

    # Load dataset for corpus
    with Timer("Dataset loading"):
        data = load_dataset_splits()
        corpus = data['test']['corpus']
        sample = args.sample if (hasattr(args, 'sample') and args.sample) else None
        if sample:
            corpus = dict(list(corpus.items())[:sample])
            logger.info(f"Sampled corpus down to {len(corpus)} documents")

    # Initialize pipeline
    with Timer("Pipeline initialization"):
        pipeline = RetrievalPipeline(config)
        # Build or load indexes
        pipeline.build_index(corpus, version=f"{version}_sample_{sample}" if sample else version)

    # Run retrieval
    start_time = time.time()
    results = pipeline.retrieve(
        query=query,
        top_k=top_k,
        version=version,
        use_reranker=use_reranker,
        method=method
    )
    total_time = time.time() - start_time

    # Display results
    logger.info("=" * 60)
    logger.info(f"RESULTS (retrieved in {total_time:.3f}s)")
    logger.info("=" * 60)

    format_results(results, corpus, logger)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval query")
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--version", type=str, default="default", help="Version to search")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--method", type=str, choices=["semantic", "lexical", "hybrid"],
                         default="hybrid", help="Retrieval method")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    args = parser.parse_args()
    run_retrieve(args)
