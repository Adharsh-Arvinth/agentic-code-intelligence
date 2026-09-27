"""Automated experiment runner for comparing retrieval methods."""

import json
import os
import time
import logging
from typing import Dict, List, Any, Optional

from src.evaluation.metrics import evaluate_retrieval

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run and compare retrieval experiments."""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = getattr(config, 'results_dir', 'results') if not isinstance(config, dict) else config.get('results_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.experiments: List[Dict] = []
    
    def run_experiment(
        self, 
        name: str,
        retrieval_method: str = 'hybrid',
        qrels: Optional[Dict] = None,
        results_dict: Optional[Dict] = None,
        k_values: Optional[List[int]] = None
    ) -> Dict:
        """Run a single experiment."""
        if k_values is None:
            k_values = [1, 5, 10, 100]
        
        if qrels is None or results_dict is None:
            # Load dataset and run retrieval
            logger.info(f"Running experiment '{name}' with method '{retrieval_method}'...")
            qrels, results_dict = self._run_retrieval(retrieval_method)
        
        logger.info(f"Evaluating experiment: {name}")
        start = time.time()
        metrics = evaluate_retrieval(qrels, results_dict, k_values)
        eval_time = time.time() - start
        
        experiment_result = {
            'name': name,
            'method': retrieval_method,
            'metrics': metrics,
            'eval_time': eval_time,
            'num_queries': len(qrels),
            'num_results': len(results_dict)
        }
        self.experiments.append(experiment_result)
        
        logger.info(f"Experiment '{name}': NDCG@10={metrics.get('ndcg@10', 0):.4f}, MRR={metrics.get('mrr', 0):.4f}")
        return experiment_result
    
    def _run_retrieval(self, method: str) -> tuple:
        """Run retrieval pipeline and return qrels and results."""
        from src.utils.data_loader import load_dataset_splits
        from src.retrieval.pipeline import RetrievalPipeline
        
        # Load data
        data = load_dataset_splits()
        test_data = data.get('test', {})
        queries = test_data.get('queries', {})
        corpus = test_data.get('corpus', {})
        qrels = test_data.get('qrels', {})
        
        max_queries = getattr(self.config, 'max_queries', None)
        if max_queries and len(queries) > max_queries:
            queries = dict(list(queries.items())[:max_queries])
            qrels = {qid: qrels[qid] for qid in queries if qid in qrels}
            logger.info(f"Limited evaluation to {len(queries)} queries")
        
        if not queries or not corpus:
            raise ValueError("No test data available. Ensure the dataset is accessible.")
        
        # Initialize pipeline
        pipeline = RetrievalPipeline(self.config)
        pipeline.build_index(corpus, version='default')
        
        # Run retrieval
        results_dict = {}
        total = len(queries)
        logger.info(f"Running retrieval for {total} queries using method '{method}'...")
        
        for i, (qid, query_text) in enumerate(queries.items()):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{total}")
            
            try:
                results = pipeline.retrieve(
                    query=query_text,
                    top_k=100,
                    version='default',
                    use_reranker=(method == 'hybrid_rerank'),
                    method='hybrid' if 'hybrid' in method else method
                )
                # Convert to ranked doc_id list
                results_dict[qid] = [r['doc_id'] for r in results]
            except Exception as e:
                logger.warning(f"Query {qid} failed: {e}")
                results_dict[qid] = []
        
        return qrels, results_dict
    
    def run_all_experiments(self) -> List[Dict]:
        """Run all standard experiments."""
        experiments_to_run = [
            ('Semantic Only', 'semantic'),
            ('Lexical Only', 'lexical'),
            ('Hybrid (RRF)', 'hybrid'),
        ]
        
        for name, method in experiments_to_run:
            try:
                self.run_experiment(name=name, retrieval_method=method)
            except Exception as e:
                logger.error(f"Experiment '{name}' failed: {e}")
                self.experiments.append({
                    'name': name,
                    'method': method,
                    'metrics': {},
                    'error': str(e)
                })
        
        return self.experiments
    
    def save_results(self, results: Any, path: str):
        """Save results to JSON."""
        full_path = os.path.join(self.results_dir, path) if not os.path.isabs(path) else path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {full_path}")
    
    def compare_experiments(self, results: Any = None) -> str:
        """Format comparison table of experiment results."""
        if results is None:
            results = self.experiments
        
        if isinstance(results, dict):
            results = [results]
        
        if not results:
            return "No experiments to compare."
        
        header = f"{'Experiment':<25} | {'NDCG@10':<10} | {'MRR':<10} | {'Recall@10':<10}"
        separator = "-" * len(header)
        lines = ["\n" + separator, header, separator]
        
        for exp in results:
            if isinstance(exp, dict) and 'metrics' in exp:
                name = str(exp.get('name', 'Unknown'))[:25]
                metrics = exp['metrics']
                ndcg = metrics.get('ndcg@10', 0.0)
                mrr_val = metrics.get('mrr', 0.0)
                recall = metrics.get('recall@10', 0.0)
                lines.append(f"{name:<25} | {ndcg:<10.4f} | {mrr_val:<10.4f} | {recall:<10.4f}")
            elif isinstance(exp, dict) and 'error' in exp:
                name = str(exp.get('name', 'Unknown'))[:25]
                lines.append(f"{name:<25} | ERROR: {exp['error'][:30]}")
        
        lines.append(separator)
        return "\n".join(lines)
