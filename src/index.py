"""Build indexes for the code corpus."""

import argparse
import time
import os
import sys


def run_index(args):
    """Build embedding and lexical indexes for the corpus."""
    from src.utils.config import load_config
    from src.utils.data_loader import load_dataset_splits
    from src.utils.logging_utils import setup_logger, Timer
    from src.indexing.index_manager import IndexManager

    logger = setup_logger("index")
    config = load_config(args.config if hasattr(args, 'config') else "configs/default.yaml")

    version = args.version if hasattr(args, 'version') else "default"
    force = args.force if hasattr(args, 'force') else False

    logger.info("=" * 60)
    logger.info("AGENTIC CODE INTELLIGENCE - INDEX BUILDER")
    logger.info("=" * 60)

    # Load dataset
    with Timer("Dataset loading"):
        data = load_dataset_splits()
        corpus = data['test']['corpus']
        logger.info(f"Loaded corpus with {len(corpus)} documents")

    # Build indexes
    index_manager = IndexManager(config)

    if index_manager.is_indexed(version) and not force:
        logger.info(f"Indexes already exist for version '{version}'. Use --force to rebuild.")
        return

    with Timer("Index building"):
        index_manager.build_all(corpus, version=version)

    logger.info("=" * 60)
    logger.info("INDEXING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build indexes")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--version", type=str, default="default")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_index(args)
