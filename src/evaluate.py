"""Run custom evaluation experiments."""

import argparse
import json
import os


def run_evaluate(args):
    """Run evaluation experiments comparing different retrieval methods."""
    from src.utils.config import load_config
    from src.utils.logging_utils import setup_logger, Timer
    from src.evaluation.experiment_runner import ExperimentRunner

    logger = setup_logger("evaluate")
    config = load_config(args.config if hasattr(args, 'config') else "configs/default.yaml")
    output_dir = args.output if hasattr(args, 'output') else "experiments/"
    experiments = args.experiments if hasattr(args, 'experiments') else "all"

    logger.info("=" * 60)
    logger.info("AGENTIC CODE INTELLIGENCE - EVALUATION")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    runner = ExperimentRunner(config)

    if experiments == "all":
        results = runner.run_all_experiments()
    else:
        results = runner.run_experiment(experiments)

    # Save results
    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Print comparison
    comparison = runner.compare_experiments(results)
    logger.info("\n" + comparison)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation experiments")
    parser.add_argument("--experiments", type=str, default="all",
                         choices=["all", "lexical", "semantic", "hybrid", "hybrid_rerank"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="experiments/")
    args = parser.parse_args()
    run_evaluate(args)
