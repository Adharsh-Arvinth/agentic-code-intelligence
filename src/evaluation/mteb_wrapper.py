"""MTEB-compatible model wrapper for code retrieval evaluation."""

import logging
import numpy as np
from typing import List, Dict, Union

logger = logging.getLogger(__name__)


class CodeRetrievalModel:
    """
    MTEB-compatible embedding model wrapper for code retrieval.
    
    Implements encode(), encode_queries(), and encode_corpus() methods
    required by the MTEB evaluation framework.
    """
    
    def __init__(self, model_name='jinaai/jina-embeddings-v2-base-code', device='auto'):
        try:
            import torch
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
        
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading model: {model_name} on {device}")
        
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name, 
            device=self.device,
            trust_remote_code=True
        )
        
        # Set query instruction based on model type
        self.query_instruction = ""
        if "instructor" in model_name.lower():
            self.query_instruction = "Represent the code query for retrieval: "
        elif "e5" in model_name.lower() and "instruct" in model_name.lower():
            self.query_instruction = "query: "
    
    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode a list of sentences into embeddings."""
        if not sentences:
            return np.array([])
        
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=kwargs.get('show_progress_bar', len(sentences) > 100),
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def encode_queries(self, queries: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode queries with optional instruction prefix."""
        if not queries:
            return np.array([])
        
        if self.query_instruction:
            formatted_queries = [self.query_instruction + q for q in queries]
        else:
            formatted_queries = queries
        
        return self.encode(formatted_queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(
        self, 
        corpus: Union[List[Dict[str, str]], List[str]], 
        batch_size: int = 32, 
        **kwargs
    ) -> np.ndarray:
        """
        Encode corpus documents.
        
        Handles both MTEB format (list of dicts with 'title' and 'text')
        and plain string lists.
        """
        if not corpus:
            return np.array([])
        
        texts = []
        if isinstance(corpus[0], dict):
            for doc in corpus:
                title = doc.get('title', '').strip()
                text = doc.get('text', '').strip()
                if title:
                    texts.append(f"{title}\n{text}")
                else:
                    texts.append(text)
        elif isinstance(corpus[0], str):
            texts = list(corpus)
        else:
            texts = [str(doc) for doc in corpus]
        
        return self.encode(texts, batch_size=batch_size, **kwargs)
