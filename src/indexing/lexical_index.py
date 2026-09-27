import re
import pickle
import numpy as np
import math
from typing import List, Tuple, Dict
from collections import Counter

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

class SimpleTFIDF:
    def __init__(self, corpus):
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        df = Counter()
        self.N = len(corpus)
        for doc in corpus:
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            self.doc_len.append(len(doc))
            for word in freq:
                df[word] += 1
        
        for word, freq in df.items():
            self.idf[word] = math.log(self.N / (freq + 0.5))

    def get_scores(self, query):
        scores = np.zeros(self.N)
        for q in query:
            q_idf = self.idf.get(q, 0)
            if q_idf == 0:
                continue
            for i in range(self.N):
                freq = self.doc_freqs[i].get(q, 0)
                scores[i] += q_idf * (freq / (freq + 1))
        return scores

class LexicalIndex:
    def __init__(self):
        self.model = None
        self.doc_ids = []

    def tokenize(self, text: str) -> List[str]:
        tokens = re.sub('([a-z])([A-Z])', r'\1 \2', text).replace('_', ' ')
        tokens = re.findall(r'\b\w+\b', tokens)
        return [t.lower() for t in tokens]

    def build_index(self, documents: List[str], doc_ids: List[str]):
        self.doc_ids = doc_ids
        tokenized_corpus = [self.tokenize(doc) for doc in documents]
        if BM25Okapi is not None:
            self.model = BM25Okapi(tokenized_corpus)
        else:
            self.model = SimpleTFIDF(tokenized_corpus)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        tokenized_query = self.tokenize(query)
        scores = self.model.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    def batch_search(self, queries: List[str], top_k: int) -> Dict[int, List[Tuple[str, float]]]:
        results = {}
        for i, q in enumerate(queries):
            results[i] = self.search(q, top_k)
        return results

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'doc_ids': self.doc_ids}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.doc_ids = data['doc_ids']
