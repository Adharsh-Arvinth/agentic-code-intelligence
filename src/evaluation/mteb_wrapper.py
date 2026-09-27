"""MTEB-compatible model wrapper for code retrieval evaluation."""

import logging
import numpy as np
from typing import List, Dict, Union, Any

logger = logging.getLogger(__name__)

try:
    from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
    BaseWrapper = SentenceTransformerEncoderWrapper
except ImportError:
    BaseWrapper = object


class CodeRetrievalModel(BaseWrapper):
    """
    MTEB-compatible embedding model wrapper for code retrieval.
    
    Implements encode(), encode_queries(), and encode_corpus() methods
    required by the MTEB evaluation framework.
    """
    
    def __init__(self, model_name='BAAI/bge-small-en-v1.5', device='auto'):
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
        
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading model: {model_name} on {device}")
        st_model = SentenceTransformer(
            model_name, 
            device=self.device
        )
        
        if BaseWrapper is not object:
            super().__init__(st_model)
            self.model = st_model
        else:
            self.model = st_model
        
        # Set query instruction based on model type
        self.query_instruction = ""
        if "instructor" in model_name.lower():
            self.query_instruction = "Represent the code query for retrieval: "
        elif "e5" in model_name.lower() and "instruct" in model_name.lower():
            self.query_instruction = "query: "
    
    def encode(self, sentences: Any, batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode sentences or DataLoader into embeddings."""
        if sentences is None:
            return np.array([])
        
        # Unwrap DataLoader / Generator / Iterable if needed
        if not isinstance(sentences, (list, tuple, np.ndarray)):
            items = []
            for batch in sentences:
                if isinstance(batch, (list, tuple)):
                    items.extend(batch)
                elif isinstance(batch, dict) and 'text' in batch:
                    txt = batch['text']
                    if isinstance(txt, (list, tuple)):
                        items.extend(txt)
                    else:
                        items.append(txt)
                elif isinstance(batch, str):
                    items.append(batch)
                else:
                    items.append(batch)
            sentences = items

        if not sentences:
            return np.array([])

        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=kwargs.get('show_progress_bar', False),
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def encode_queries(self, queries: Any, batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode queries with optional instruction prefix."""
        if queries is None:
            return np.array([])
        
        if not isinstance(queries, (list, tuple, np.ndarray)):
            items = []
            for batch in queries:
                if isinstance(batch, (list, tuple)):
                    items.extend(batch)
                else:
                    items.append(batch)
            queries = items

        if self.query_instruction:
            formatted_queries = [self.query_instruction + str(q) for q in queries]
        else:
            formatted_queries = [str(q) for q in queries]
        
        return self.encode(formatted_queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(
        self, 
        corpus: Any, 
        batch_size: int = 32, 
        **kwargs
    ) -> np.ndarray:
        """
        Encode corpus documents.
        
        Handles both MTEB format (list of dicts with 'title' and 'text')
        and plain string lists.
        """
        if corpus is None:
            return np.array([])
        
        if not isinstance(corpus, (list, tuple, np.ndarray)):
            items = []
            for batch in corpus:
                if isinstance(batch, (list, tuple)):
                    items.extend(batch)
                else:
                    items.append(batch)
            corpus = items

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
