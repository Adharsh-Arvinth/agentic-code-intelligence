import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Tuple, List

class EmbeddingIndex:
    def __init__(self, model_name: str, device: str, cache_dir: str):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir, trust_remote_code=True)
        self.index = None

    def encode_documents(self, documents: List[str], batch_size: int = 128, show_progress: bool = True) -> np.ndarray:
        from src.evaluation.mteb_wrapper import PersistentEmbeddingCache
        cache = PersistentEmbeddingCache(self.model_name, 256, os.path.join(self.cache_dir, "emb_cache"))
        embs = cache.encode_with_cache(self.model, documents, batch_size=batch_size, max_len_override=256)
        cache.save()
        return embs.astype(np.float32)

    def encode_queries(self, queries: List[str], batch_size: int = 128) -> np.ndarray:
        from src.preprocessing.representations import format_query_smart
        formatted = [format_query_smart(q, self.model_name) for q in queries]
        embeddings = self.model.encode(
            formatted, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def build_index(self, embeddings: np.ndarray):
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        if self.device.startswith('cuda') and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        return scores[0], indices[0]

    def batch_search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices

    def save(self, path: str):
        index_to_save = self.index
        if hasattr(faiss, 'index_gpu_to_cpu') and type(self.index) != faiss.IndexFlatIP:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(index_to_save, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
        if self.device.startswith('cuda') and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def save_embeddings(self, embeddings: np.ndarray, path: str):
        np.save(path, embeddings)

    def load_embeddings(self, path: str) -> np.ndarray:
        return np.load(path)
