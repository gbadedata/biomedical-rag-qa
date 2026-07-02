"""Retrievers over a passage corpus.

Three implementations behind one interface so they can be swapped and benchmarked:
  - BM25Retriever   : lexical baseline (rank_bm25).
  - DenseRetriever  : TF-IDF reduced by truncated SVD (latent semantic analysis),
                      L2-normalised and indexed in FAISS (inner product == cosine).
  - RandomRetriever : shuffled order, the floor any real retriever must beat.

The dense retriever is deliberately swappable: replacing the vectoriser with a
biomedical transformer embedder (e.g. PubMedBERT or an embedding API) is a drop-in
change to `._embed`, and the FAISS index and search loop stay the same.
"""
from __future__ import annotations

import random
from typing import Protocol

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .corpus import Passage, tokenize


class Retriever(Protocol):
    name: str

    def index(self, passages: list[Passage]) -> "Retriever": ...

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []

    def index(self, passages: list[Passage]) -> "BM25Retriever":
        self._ids = [p.passage_id for p in passages]
        self._bm25 = BM25Okapi([tokenize(p.text) for p in passages])
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        assert self._bm25 is not None, "call index() first"
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in top]


class DenseRetriever:
    name = "dense_lsa_faiss"

    def __init__(self, n_components: int = 256, ngram_range=(1, 2), min_df: int = 2,
                 random_state: int = 42) -> None:
        self.n_components = n_components
        self._vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df,
                                    sublinear_tf=True, stop_words="english")
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._index: faiss.Index | None = None
        self._ids: list[str] = []

    def _embed(self, texts: list[str], fit: bool = False) -> np.ndarray:
        if fit:
            tfidf = self._vec.fit_transform(texts)
            # cap components below the feature count to keep SVD valid on small corpora
            self._svd.n_components = min(self.n_components, tfidf.shape[1] - 1)
            dense = self._svd.fit_transform(tfidf)
        else:
            dense = self._svd.transform(self._vec.transform(texts))
        return normalize(dense.astype("float32"))

    def index(self, passages: list[Passage]) -> "DenseRetriever":
        self._ids = [p.passage_id for p in passages]
        mat = self._embed([p.text for p in passages], fit=True)
        self._index = faiss.IndexFlatIP(mat.shape[1])
        self._index.add(mat)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        assert self._index is not None, "call index() first"
        q = self._embed([query], fit=False)
        scores, idx = self._index.search(q, k)
        return [(self._ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]


class RandomRetriever:
    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self._ids: list[str] = []
        self._rng = random.Random(seed)

    def index(self, passages: list[Passage]) -> "RandomRetriever":
        self._ids = [p.passage_id for p in passages]
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        pool = self._ids[:]
        self._rng.shuffle(pool)
        return [(pid, 0.0) for pid in pool[:k]]
