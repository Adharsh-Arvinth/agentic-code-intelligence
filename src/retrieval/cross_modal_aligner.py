"""
Cross-Modal Contrastive Subspace Aligner for Text-to-Code Retrieval.

Learns an Identity-Anchored Closed-Form Ridge / Procrustes Linear Transformation
mapping natural-language competitive programming problem embeddings into the
Python code solution subspace using ONLY the official `train` split
(strictly disjoint from validation and test splits).
"""

import os
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


class CrossModalAligner:
    """
    Identity-Anchored Ridge Alignment:
        W = (Q^T Q + reg * I)^(-1) Q^T D
        z_Q = Normalize((1 - alpha) * u_Q + alpha * (u_Q @ W))
    Guarantees preservation of pretrained embedding geometry while correcting
    systematic modality drift between English math specifications and Python code.
    """

    def __init__(self, embed_dim: int = 384, struct_dim: int = 384, out_dim: int = 384, reg: float = 150.0, alpha: float = 0.22):
        self.embed_dim = embed_dim
        self.struct_dim = struct_dim
        self.out_dim = out_dim
        self.reg = reg
        self.alpha = alpha
        self.W: Optional[np.ndarray] = None
        self.is_trained = False

    def fit(
        self,
        train_query_dense: np.ndarray,
        train_query_texts: List[str],
        train_doc_dense: np.ndarray,
        train_doc_texts: List[str],
        **kwargs,
    ) -> None:
        """Compute closed-form ridge alignment matrix W on training pairs."""
        Q = train_query_dense.astype(np.float64)
        D = train_doc_dense.astype(np.float64)
        dim = Q.shape[1]
        gram = Q.T @ Q + self.reg * np.eye(dim, dtype=np.float64)
        cross = Q.T @ D
        self.W = np.linalg.solve(gram, cross).astype(np.float32)
        self.is_trained = True
        logger.info(f"CrossModalAligner fitted closed-form Ridge transformation ({dim}x{dim}, reg={self.reg})")

    def project_queries(self, dense_embs: np.ndarray, formatted_texts: Optional[List[str]] = None) -> np.ndarray:
        if not self.is_trained or self.W is None:
            return dense_embs
        mapped = dense_embs @ self.W
        m_norms = np.linalg.norm(mapped, axis=1, keepdims=True)
        m_norms[m_norms == 0] = 1.0
        mapped = mapped / m_norms

        blended = (1.0 - self.alpha) * dense_embs + self.alpha * mapped
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (blended / norms).astype(np.float32)

    def project_corpus(self, dense_embs: np.ndarray, formatted_texts: Optional[List[str]] = None) -> np.ndarray:
        return dense_embs

    def save(self, path: str) -> None:
        if self.W is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path + ".npz", W=self.W, reg=self.reg, alpha=self.alpha)

    def load(self, path: str) -> bool:
        npz_path = path + ".npz" if not path.endswith(".npz") else path
        if not os.path.exists(npz_path):
            return False
        try:
            data = np.load(npz_path)
            self.W = data["W"].astype(np.float32)
            self.reg = float(data["reg"])
            self.alpha = float(data["alpha"])
            self.is_trained = True
            return True
        except Exception:
            return False
