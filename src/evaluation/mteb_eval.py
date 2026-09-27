"""Run official MTEB evaluation for code retrieval models."""

import argparse
import os
import sys
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MTEB evaluation for code retrieval models")
    parser.add_argument("--model", type=str, default="jinaai/jina-embeddings-v2-base-code",
                        help="HuggingFace model name")
    parser.add_argument("--output_path", type=str, default="results/mteb_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    args = parser.parse_args()

    try:
        import mteb
    except ImportError:
        print("Error: mteb not installed. Run: pip install mteb")
        sys.exit(1)

    from src.evaluation.mteb_wrapper import CodeRetrievalModel

    os.makedirs(args.output_path, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = CodeRetrievalModel(model_name=args.model)

    print("Loading AppsRetrieval task...")
    tasks = mteb.get_tasks(tasks=["AppsRetrieval"])

    evaluation = mteb.MTEB(tasks=tasks)

    print("Running evaluation...")
    results = evaluation.run(
        model,
        output_folder=args.output_path,
        eval_splits=["test"],
        overwrite_results=True,
        encode_kwargs={"batch_size": args.batch_size}
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for task_result in results:
        task_name = getattr(task_result, "task_name", "AppsRetrieval")
        scores = getattr(task_result, "scores", {})
        test_metrics = scores.get("test", [{}])
        if isinstance(test_metrics, list) and test_metrics:
            test_metrics = test_metrics[0]

        ndcg_10 = test_metrics.get("ndcg_at_10", 0.0)
        mrr_10 = test_metrics.get("mrr_at_10", 0.0)
        recall_10 = test_metrics.get("recall_at_10", 0.0)

        print(f"Task: {task_name}")
        print(f"  NDCG@10:   {ndcg_10:.4f}")
        print(f"  MRR@10:    {mrr_10:.4f}")
        print(f"  Recall@10: {recall_10:.4f}")

    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
