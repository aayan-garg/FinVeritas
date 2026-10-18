"""
FinVeritas: Lexical Retrieval Baselines for Financial Documents.

Implements two CPU-friendly page-level retrieval engines:
1. Word TF-IDF with cosine similarity.
2. BM25 (Okapi BM25 with financial tokenization and inverted indexing).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from finveritas.rag_indexing.financebench_data import PageCorpusItem


@dataclasses.dataclass(frozen=True)
class RetrievalResult:
    """Represents a single retrieved page candidate."""

    page_id: str
    doc_name: str
    page_index: int
    score: float
    rank: int  # 1-based rank
    text_snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "doc_name": self.doc_name,
            "page_index": self.page_index,
            "score": round(self.score, 6),
            "rank": self.rank,
            "text_snippet": self.text_snippet,
        }


# Tokenizer regex preserving alphanumeric words, dollar signs, percentages, and hyphenated terms
FIN_TOKEN_RE = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b|[$%]")


def tokenize_financial_query(text: str) -> list[str]:
    """Tokenizes financial queries preserving numbers, symbols, and hyphenations."""
    return [tok.lower() for tok in FIN_TOKEN_RE.findall(text) if len(tok) > 0]


class TfidfRetriever:
    """Page-level lexical retriever using Word TF-IDF and Cosine Similarity."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b|[$%]",
            lowercase=True,
        )
        self.corpus: list[PageCorpusItem] = []
        self.tfidf_matrix = None
        self.is_fitted = False

    def fit(self, corpus: list[PageCorpusItem]) -> "TfidfRetriever":
        """Fits the TF-IDF vectorizer and transforms the entire corpus."""
        if not corpus:
            raise ValueError("Cannot fit TfidfRetriever on an empty corpus.")

        self.corpus = corpus
        corpus_texts = [item.text for item in corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.is_fitted = True
        return self

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieves the top_k pages for a given query."""
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("TfidfRetriever must be fitted before retrieval.")

        query_vec = self.vectorizer.transform([query])
        # Cosine similarity for L2-normalized TF-IDF reduces to dot product
        scores = linear_kernel(query_vec, self.tfidf_matrix).flatten()

        n_docs = len(scores)
        k = min(top_k, n_docs)
        if k == 0:
            return []

        # Get top-k indices sorted descending by score
        if n_docs > k:
            candidate_indices = np.argpartition(scores, -k)[-k:]
            sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-scores)

        results: list[RetrievalResult] = []
        for rank_idx, doc_idx in enumerate(sorted_indices, start=1):
            item = self.corpus[int(doc_idx)]
            score = float(scores[doc_idx])
            snippet = item.text[:300].replace("\n", " ").strip()
            results.append(
                RetrievalResult(
                    page_id=item.page_id,
                    doc_name=item.doc_name,
                    page_index=item.page_index,
                    score=score,
                    rank=rank_idx,
                    text_snippet=snippet,
                )
            )

        return results


class BM25Retriever:
    """
    Page-level lexical retriever using Okapi BM25 with financial tokenization.
    Implements inverted index scoring for ultra-fast CPU retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus: list[PageCorpusItem] = []
        self.doc_len: np.ndarray = np.array([])
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        # Inverted index: term -> list of (doc_idx, term_freq)
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.is_fitted = False

    def fit(self, corpus: list[PageCorpusItem]) -> "BM25Retriever":
        """Builds the BM25 inverted index and calculates Robertson-Spärck Jones IDF."""
        if not corpus:
            raise ValueError("Cannot fit BM25Retriever on an empty corpus.")

        self.corpus = corpus
        num_docs = len(corpus)
        doc_lengths = []
        doc_frequencies: dict[str, int] = Counter()
        inverted_index: dict[str, list[tuple[int, int]]] = {}

        for doc_idx, item in enumerate(corpus):
            tokens = tokenize_financial_query(item.text)
            doc_len = len(tokens)
            doc_lengths.append(doc_len)

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                doc_frequencies[term] += 1
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((doc_idx, count))

        self.doc_len = np.array(doc_lengths, dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if num_docs > 0 else 0.0
        self.inverted_index = inverted_index

        # Compute Robertson-Spärck Jones IDF with floor threshold
        idf_dict: dict[str, float] = {}
        negative_idfs = []
        for term, df in doc_frequencies.items():
            # Standard Lucene/BM25 IDF formula: ln(1 + (N - df + 0.5) / (df + 0.5))
            val = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            idf_dict[term] = val
            if val < 0:
                negative_idfs.append(term)

        # Handle edge cases where IDF could be negative for terms in >50% docs
        if negative_idfs:
            eps_val = self.epsilon * (sum(idf_dict.values()) / max(len(idf_dict), 1))
            for term in negative_idfs:
                idf_dict[term] = eps_val

        self.idf = idf_dict
        self.is_fitted = True
        return self

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieves the top_k pages for a query using accumulated inverted index scoring."""
        if not self.is_fitted or len(self.corpus) == 0:
            raise RuntimeError("BM25Retriever must be fitted before retrieval.")

        query_tokens = tokenize_financial_query(query)
        if not query_tokens:
            return []

        num_docs = len(self.corpus)
        scores = np.zeros(num_docs, dtype=np.float32)

        # Precompute denominator normalization factor: k1 * (1 - b + b * (doc_len / avgdl))
        denom_norm = self.k1 * (1.0 - self.b + self.b * (self.doc_len / self.avgdl))

        query_term_counts = Counter(query_tokens)
        for term, q_tf in query_term_counts.items():
            if term not in self.inverted_index:
                continue

            term_idf = self.idf.get(term, 0.0)
            if term_idf <= 0.0:
                continue

            postings = self.inverted_index[term]
            doc_indices = np.array([p[0] for p in postings], dtype=np.int32)
            tfs = np.array([p[1] for p in postings], dtype=np.float32)

            # BM25 term saturation formula: IDF * (tf * (k1 + 1)) / (tf + denom_norm)
            term_scores = term_idf * (tfs * (self.k1 + 1.0)) / (tfs + denom_norm[doc_indices])
            scores[doc_indices] += term_scores

        k = min(top_k, num_docs)
        if k == 0:
            return []

        # Find top-k documents
        if num_docs > k:
            candidate_indices = np.argpartition(scores, -k)[-k:]
            sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-scores)

        results: list[RetrievalResult] = []
        for rank_idx, doc_idx in enumerate(sorted_indices, start=1):
            score = float(scores[doc_idx])
            item = self.corpus[int(doc_idx)]
            snippet = item.text[:300].replace("\n", " ").strip()
            results.append(
                RetrievalResult(
                    page_id=item.page_id,
                    doc_name=item.doc_name,
                    page_index=item.page_index,
                    score=score,
                    rank=rank_idx,
                    text_snippet=snippet,
                )
            )

        return results
