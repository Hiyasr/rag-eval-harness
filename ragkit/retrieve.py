"""BM25 + dense (FAISS) + reciprocal-rank fusion + optional reranking."""
from __future__ import annotations
import math
from collections import Counter
import faiss
import numpy as np
from .chunking import Chunk
from .embed import tokenize


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.docs), dtype="float32")
        for t in set(query):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / max(self.avg, 1e-9))
                    s[i] += self.idf[t] * f * (self.k1 + 1) / denom
        return s


class OverlapReranker:
    """Cheap offline reranker: query-term coverage. Replace with CrossEncoderReranker."""
    name = "overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / max(len(q), 1) for t in texts]


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.name = model
        self.model = CrossEncoder(model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self.model.predict([(query, t) for t in texts])]


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder, mode: str = "hybrid", reranker=None,
                 candidates: int = 20, rrf_k: int = 60):
        assert mode in {"bm25", "dense", "hybrid"}
        self.chunks, self.embedder, self.mode = chunks, embedder, mode
        self.reranker, self.candidates, self.rrf_k = reranker, candidates, rrf_k
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        vecs = embedder.encode([c.text for c in chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def _bm25_rank(self, q: str, n: int) -> list[int]:
        s = self.bm25.scores(tokenize(q))
        return [int(i) for i in np.argsort(-s)[:n] if s[i] > 0]

    def _dense_rank(self, q: str, n: int) -> list[int]:
        _, idx = self.index.search(self.embedder.encode([q]), min(n, len(self.chunks)))
        return [int(i) for i in idx[0] if i >= 0]

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        n = max(self.candidates, k)
        if self.mode == "bm25":
            ranked = self._bm25_rank(query, n)
        elif self.mode == "dense":
            ranked = self._dense_rank(query, n)
        else:
            fused: dict[int, float] = {}
            for ranking in (self._bm25_rank(query, n), self._dense_rank(query, n)):
                for r, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
            ranked = sorted(fused, key=fused.get, reverse=True)[:n]
        if self.reranker and ranked:
            sc = self.reranker.score(query, [self.chunks[i].text for i in ranked])
            ranked = [i for _, i in sorted(zip(sc, ranked), key=lambda x: -x[0])]
        return [self.chunks[i] for i in ranked[:k]]
