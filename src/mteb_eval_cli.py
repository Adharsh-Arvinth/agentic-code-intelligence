"""Run official MTEB AppsRetrieval evaluation."""

import argparse
import os
import json
import sys


def run_mteb_eval(args=None):
    """Run the official MTEB evaluation on AppsRetrieval."""
    from src.utils.config import load_config
    from src.utils.logging_utils import setup_logger, Timer

    logger = setup_logger("mteb_eval")

    if args is None:
        parser = argparse.ArgumentParser(description="MTEB AppsRetrieval Evaluation")
        parser.add_argument("--model", type=str, default=None, help="Embedding model name")
        parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
        parser.add_argument("--output", type=str, default="results/", help="Output directory")
        parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
        args = parser.parse_args()

    config = load_config(args.config if hasattr(args, 'config') else "configs/default.yaml")

    model_name = args.model if (hasattr(args, 'model') and args.model) else config.embedding_model
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 64
    output_dir = args.output if hasattr(args, 'output') else "results/"

    logger.info("=" * 60)
    logger.info("MTEB APPSRETRIEVAL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {config.get_device()}")
    logger.info("-" * 60)

    try:
        import mteb
    except ImportError:
        logger.error("mteb not installed. Run: pip install mteb")
        sys.exit(1)

    # Import and create model wrapper
    from src.evaluation.mteb_wrapper import CodeRetrievalModel

    with Timer("Model loading"):
        model = CodeRetrievalModel(
            model_name=model_name,
            device=config.get_device()
        )

    logger.info("Loading AppsRetrieval task...")
    tasks = mteb.get_tasks(tasks=["AppsRetrieval"])

    logger.info("Starting MTEB evaluation...")
    evaluation = mteb.MTEB(tasks=tasks)

    os.makedirs(output_dir, exist_ok=True)

    with Timer("MTEB evaluation"):
        results = evaluation.run(
            model,
            output_folder=output_dir,
            eval_splits=["test"],
            overwrite_results=True,
            encode_kwargs={"batch_size": batch_size}
        )

    # Print key metrics
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    if results:
        for task_result in results:
            # Try to extract metrics from result object
            try:
                # MTEB returns result objects - extract scores
                if hasattr(task_result, 'scores'):
                    test_scores = task_result.scores.get('test', [{}])
                    if test_scores:
                        scores = test_scores[0] if isinstance(test_scores, list) else test_scores
                        ndcg_10 = scores.get('ndcg_at_10', 'N/A')
                        mrr_10 = scores.get('mrr_at_10', 'N/A')
                        recall_10 = scores.get('recall_at_10', 'N/A')
                        map_10 = scores.get('map_at_10', 'N/A')

                        logger.info(f"NDCG@10:   {ndcg_10}")
                        logger.info(f"MRR@10:    {mrr_10}")
                        logger.info(f"Recall@10: {recall_10}")
                        logger.info(f"MAP@10:    {map_10}")
                else:
                    logger.info(f"Raw result: {task_result}")
            except Exception as e:
                logger.info(f"Result: {task_result}")
                logger.warning(f"Could not parse result details: {e}")

    # Find and report the output JSON
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.json') and 'Apps' in f:
                json_path = os.path.join(root, f)
                logger.info(f"Results JSON: {json_path}")
                try:
                    with open(json_path, 'r') as jf:
                        result_data = json.load(jf)
                    # Extract key metrics from saved JSON
                    if 'scores' in result_data:
                        test_data = result_data['scores'].get('test', [{}])
                        if test_data:
                            s = test_data[0] if isinstance(test_data, list) else test_data
                            logger.info(f"  NDCG@10: {s.get('ndcg_at_10', 'N/A')}")
                            logger.info(f"  MRR@10:  {s.get('mrr_at_10', 'N/A')}")
                except Exception:
                    pass

    logger.info("=" * 60)
    logger.info("MTEB EVALUATION COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_mteb_eval()
