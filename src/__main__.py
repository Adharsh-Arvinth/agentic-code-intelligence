"""
Agentic Code Intelligence - Main CLI dispatcher.

Usage:
    python -m src index          Build indexes for the corpus
    python -m src retrieve       Run a retrieval query
    python -m src evaluate       Run custom evaluation experiments
    python -m src mteb_eval      Run official MTEB AppsRetrieval evaluation
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="agentic-code-intelligence",
        description="Agentic Code Intelligence - Code Retrieval System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Build indexes for the corpus")
    index_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    index_parser.add_argument("--version", type=str, default="default", help="Version tag for the index")
    index_parser.add_argument("--force", action="store_true", help="Force rebuild indexes")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Run a retrieval query")
    retrieve_parser.add_argument("--query", type=str, required=True, help="Natural language query")
    retrieve_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    retrieve_parser.add_argument("--version", type=str, default="default", help="Version to search")
    retrieve_parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    retrieve_parser.add_argument("--method", type=str, choices=["semantic", "lexical", "hybrid"],
                                 default="hybrid", help="Retrieval method")
    retrieve_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    retrieve_parser.add_argument("--sample", type=int, default=None, help="Sample N documents for fast execution")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation experiments")
    eval_parser.add_argument("--experiments", type=str, default="all",
                             choices=["all", "lexical", "semantic", "hybrid", "hybrid_rerank"],
                             help="Which experiments to run")
    eval_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    eval_parser.add_argument("--output", type=str, default="experiments/", help="Output directory")
    eval_parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries")

    # MTEB eval command
    mteb_parser = subparsers.add_parser("mteb_eval", help="Run official MTEB AppsRetrieval evaluation")
    mteb_parser.add_argument("--model", type=str, default=None, help="Override embedding model")
    mteb_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    mteb_parser.add_argument("--output", type=str, default="results/", help="Output directory")
    mteb_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "index":
        from src.index import run_index
        run_index(args)
    elif args.command == "retrieve":
        from src.retrieve import run_retrieve
        run_retrieve(args)
    elif args.command == "evaluate":
        from src.evaluate import run_evaluate
        run_evaluate(args)
    elif args.command == "mteb_eval":
        from src.mteb_eval_cli import run_mteb_eval
        run_mteb_eval(args)


if __name__ == "__main__":
    main()
