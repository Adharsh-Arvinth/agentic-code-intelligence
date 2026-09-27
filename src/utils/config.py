import os
import yaml
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class Config:
    embedding_model: str = 'jinaai/jina-embeddings-v2-base-code'
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    semantic_top_k: int = 200
    lexical_top_k: int = 200
    fusion_top_k: int = 100
    rerank_top_k: int = 50
    final_top_k: int = 10
    fusion_strategy: str = 'rrf'
    rrf_k: int = 60
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    batch_size: int = 64
    device: str = 'auto'
    cache_dir: str = 'data/cache'
    index_dir: str = 'data/indexes'
    results_dir: str = 'results'

    def get_device(self) -> str:
        if self.device != 'auto':
            return self.device
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_yaml(self, yaml_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f)
            
    def update_from_args(self, args: argparse.Namespace) -> None:
        valid_fields = {f.name for f in self.__dataclass_fields__.values()}
        for k, v in vars(args).items():
            if k in valid_fields and v is not None:
                setattr(self, k, v)


def load_config(yaml_path: str = 'configs/default.yaml') -> Config:
    """Load config from YAML file, or return defaults if file doesn't exist."""
    if os.path.exists(yaml_path):
        return Config.from_yaml(yaml_path)
    return Config()
