"""
FinVeritas: Lexical Retrieval Baselines for Financial Documents.

Implements two CPU-friendly page-level retrieval engines:
1. Word TF-IDF with cosine similarity.
2. BM25 (Okapi BM25 with financial tokenization and inverted indexing).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from finveritas.rag_indexing.financebench_data import PageCorpusItem

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("finveritas.retrievers")


def check_cuda_safe() -> bool:
    """Safe check for CUDA availability without leaving memory allocated or crashing."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        x = torch.zeros(1, device="cuda")
        del x
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def _atomic_write_json(file_path: Path, data: dict[str, Any]) -> None:
    """Writes JSON data atomically via a process-specific temporary file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_name(f"{file_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(file_path)


def _atomic_save_npy(file_path: Path, arr: np.ndarray) -> None:
    """Saves a NumPy array atomically via a process-specific temporary file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_name(f"{file_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(file_path)


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


@dataclasses.dataclass(frozen=True)
class ChunkRetrievalResult:
    """Represents a retrieved chunk candidate."""

    chunk_id: str
    doc_name: str
    page_index: int
    chunk_index: int
    score: float
    rank: int
    text_snippet: str
    char_start: int = 0
    char_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "page_index": self.page_index,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 6),
            "rank": self.rank,
            "text_snippet": self.text_snippet,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def deduplicate_chunks_to_pages(
    chunks: list[ChunkRetrievalResult],
    top_k: int = 50,
) -> list[RetrievalResult]:
    """
    Collapses a ranked list of chunk candidates into unique source pages.
    Each unique (doc_name, page_index) is assigned the rank and score of its highest-ranked chunk.
    """
    seen_pages: set[tuple[str, int]] = set()
    page_results: list[RetrievalResult] = []

    for ch in chunks:
        page_key = (ch.doc_name, ch.page_index)
        if page_key not in seen_pages:
            seen_pages.add(page_key)
            page_results.append(
                RetrievalResult(
                    page_id=f"{ch.doc_name}_p{ch.page_index}",
                    doc_name=ch.doc_name,
                    page_index=ch.page_index,
                    score=ch.score,
                    rank=len(page_results) + 1,
                    text_snippet=ch.text_snippet,
                )
            )
            if len(page_results) >= top_k:
                break

    return page_results


class ChunkBM25Retriever:
    """
    Chunk-level lexical retriever using Okapi BM25 with financial tokenization.
    Indexes overlapping passage chunks and maps retrieved chunks back to source PDF pages.
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
        self.chunks: list[Any] = []
        self.doc_len: np.ndarray = np.array([])
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.is_fitted = False

    def fit(self, chunks: list[Any]) -> "ChunkBM25Retriever":
        if not chunks:
            raise ValueError("Cannot fit ChunkBM25Retriever on an empty chunk list.")

        self.chunks = chunks
        num_chunks = len(chunks)
        doc_lengths = []
        doc_frequencies: dict[str, int] = Counter()
        inverted_index: dict[str, list[tuple[int, int]]] = {}

        for c_idx, ch in enumerate(chunks):
            tokens = tokenize_financial_query(ch.text)
            doc_len = len(tokens)
            doc_lengths.append(doc_len)
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                doc_frequencies[term] += 1
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((c_idx, count))

        self.doc_len = np.array(doc_lengths, dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if num_chunks > 0 else 0.0
        self.inverted_index = inverted_index

        idf_dict: dict[str, float] = {}
        negative_idfs = []
        for term, df in doc_frequencies.items():
            val = math.log(1.0 + (num_chunks - df + 0.5) / (df + 0.5))
            idf_dict[term] = val
            if val < 0:
                negative_idfs.append(term)

        if negative_idfs:
            eps_val = self.epsilon * (sum(idf_dict.values()) / max(len(idf_dict), 1))
            for term in negative_idfs:
                idf_dict[term] = eps_val

        self.idf = idf_dict
        self.is_fitted = True
        return self

    def retrieve_chunks(self, query: str, top_k: int = 50) -> list[ChunkRetrievalResult]:
        if not self.is_fitted or not self.chunks:
            raise RuntimeError("ChunkBM25Retriever must be fitted before retrieval.")

        query_tokens = tokenize_financial_query(query)
        if not query_tokens:
            return []

        num_chunks = len(self.chunks)
        scores = np.zeros(num_chunks, dtype=np.float32)
        denom_norm = self.k1 * (1.0 - self.b + self.b * (self.doc_len / self.avgdl))

        query_term_counts = Counter(query_tokens)
        for term, _ in query_term_counts.items():
            if term not in self.inverted_index:
                continue
            term_idf = self.idf.get(term, 0.0)
            if term_idf <= 0.0:
                continue

            postings = self.inverted_index[term]
            c_indices = np.array([p[0] for p in postings], dtype=np.int32)
            tfs = np.array([p[1] for p in postings], dtype=np.float32)
            term_scores = term_idf * (tfs * (self.k1 + 1.0)) / (tfs + denom_norm[c_indices])
            scores[c_indices] += term_scores

        k = min(top_k, num_chunks)
        if k == 0:
            return []

        if num_chunks > k:
            candidate_indices = np.argpartition(scores, -k)[-k:]
            sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-scores)

        results: list[ChunkRetrievalResult] = []
        for rank_idx, c_idx in enumerate(sorted_indices, start=1):
            score = float(scores[c_idx])
            item = self.chunks[int(c_idx)]
            snippet = item.text[:300].replace("\n", " ").strip()
            results.append(
                ChunkRetrievalResult(
                    chunk_id=item.chunk_id,
                    doc_name=item.doc_name,
                    page_index=item.page_index,
                    chunk_index=item.chunk_index,
                    score=score,
                    rank=rank_idx,
                    text_snippet=snippet,
                    char_start=item.char_start,
                    char_end=item.char_end,
                )
            )

        return results

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 100) -> list[RetrievalResult]:
        """Retrieves top_k unique pages by deduplicating top candidate chunks."""
        chunks = self.retrieve_chunks(query, top_k=candidate_k)
        return deduplicate_chunks_to_pages(chunks, top_k=top_k)


class DenseRetriever:
    """
    Dense vector retriever using BAAI/bge-small-en-v1.5.
    Normalizes embeddings for exact cosine similarity via dot product.
    Includes persistent, resumable, content-hashed disk cache to prevent redundant re-computation
    and protect against kernel crashes/interruption.
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: str = "cpu",
        cache_dir: Any | None = None,
        chunking_config: dict[str, Any] | None = None,
        allow_cuda: bool = False,
        embedding_dim: int = 384,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.chunking_config = chunking_config or {}
        self.embedding_dim = embedding_dim
        self.model: Any = None
        self.chunks: list[Any] = []
        self.embeddings: np.ndarray | None = None
        self.is_fitted = False

        # Device safety: Default strictly to CPU after crash.
        # Only permit CUDA if requested, allow_cuda is True, and GPU smoke test passes.
        if device == "cuda" and allow_cuda:
            if check_cuda_safe():
                self.device = "cuda"
            else:
                logger.warning(
                    "CUDA requested but GPU smoke test failed. Defaulting safely to CPU."
                )
                self.device = "cpu"
        else:
            self.device = "cpu"

    def _compute_content_hash(self, chunks: list[Any]) -> str:
        """Computes a deterministic SHA-256 fingerprint over all chunk IDs and text content."""
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode("utf-8"))
        for c in chunks:
            hasher.update(c.chunk_id.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(c.text.encode("utf-8"))
            hasher.update(b"\x00")
        return hasher.hexdigest()

    def _get_cache_key(self, chunks: list[Any]) -> str:
        """Returns stable cache key string based on content hash and chunk count."""
        content_hash = self._compute_content_hash(chunks)
        return f"{len(chunks)}_{content_hash[:16]}"

    def _get_cache_folder(self, chunks: list[Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        cache_key = self._get_cache_key(chunks)
        return self.cache_dir / f"dense_{cache_key}"

    def _validate_batch_file(
        self,
        batch_npy_path: Path,
        meta_path: Path,
        expected_count: int,
    ) -> tuple[bool, np.ndarray | None]:
        """Validates batch file integrity, shape, numeric validity, and SHA-256 checksum."""
        if not batch_npy_path.is_file() or not meta_path.is_file():
            return False, None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("num_chunks") != expected_count:
                return False, None
            arr = np.load(str(batch_npy_path))
            if arr.shape != (expected_count, self.embedding_dim):
                return False, None
            if not np.all(np.isfinite(arr)):
                return False, None
            h = hashlib.sha256()
            with open(batch_npy_path, "rb") as bf:
                for block in iter(lambda: bf.read(65536), b""):
                    h.update(block)
            if meta.get("sha256") and meta.get("sha256") != h.hexdigest():
                return False, None
            return True, arr
        except Exception:
            return False, None

    def _ensure_model_loaded(self) -> None:
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device)

    def fit(
        self,
        chunks: list[Any],
        max_chunks: int | None = None,
        force_recompute: bool = False,
    ) -> "DenseRetriever":
        if not chunks:
            raise ValueError("Cannot fit DenseRetriever on an empty chunk list.")

        if max_chunks is not None and max_chunks > 0:
            chunks = list(chunks[:max_chunks])

        self.chunks = chunks
        total_chunks = len(chunks)
        batch_size = self.batch_size
        total_batches = math.ceil(total_chunks / batch_size)
        content_hash = self._compute_content_hash(chunks)
        cache_folder = self._get_cache_folder(chunks)

        if cache_folder is not None and not force_recompute:
            cache_folder.mkdir(parents=True, exist_ok=True)
            manifest_path = cache_folder / "manifest.json"
            embeddings_path = cache_folder / "embeddings.npy"

            if manifest_path.is_file():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)

                    matches = (
                        manifest.get("content_hash_sha256") == content_hash
                        and manifest.get("model_name") == self.model_name
                        and manifest.get("num_chunks") == total_chunks
                    )

                    if matches and manifest.get("is_complete") and embeddings_path.is_file():
                        loaded = np.load(str(embeddings_path))
                        if (
                            loaded.shape == (total_chunks, self.embedding_dim)
                            and np.all(np.isfinite(loaded))
                        ):
                            self.embeddings = loaded
                            self.is_fitted = True
                            logger.info(
                                f"Loaded complete embeddings from cache: {embeddings_path} "
                                f"(N={total_chunks}, dim={self.embedding_dim})."
                            )
                            return self
                except Exception as e:
                    logger.warning(
                        f"Error reading cache manifest: {e}. Inspecting incremental batches."
                    )

        completed_batches: set[int] = set()
        manifest_data: dict[str, Any] = {}
        if cache_folder is not None:
            cache_folder.mkdir(parents=True, exist_ok=True)
            manifest_path = cache_folder / "manifest.json"

            for b in range(total_batches):
                b_start = b * batch_size
                b_end = min(b_start + batch_size, total_chunks)
                expected_count = b_end - b_start
                batch_file = cache_folder / f"batch_{b:05d}.npy"
                meta_file = cache_folder / f"batch_{b:05d}.meta.json"

                is_valid, _ = self._validate_batch_file(batch_file, meta_file, expected_count)
                if is_valid:
                    completed_batches.add(b)
                else:
                    if batch_file.is_file():
                        try:
                            batch_file.unlink()
                        except OSError:
                            pass
                    if meta_file.is_file():
                        try:
                            meta_file.unlink()
                        except OSError:
                            pass

            if completed_batches:
                logger.info(
                    f"Resuming dense embedding: {len(completed_batches)}/{total_batches} "
                    f"batches verified on disk."
                )

            manifest_data = {
                "version": "1.0",
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "device": self.device,
                "num_chunks": total_chunks,
                "content_hash_sha256": content_hash,
                "chunking_config": self.chunking_config,
                "batch_size": batch_size,
                "total_batches": total_batches,
                "completed_batches": sorted(completed_batches),
                "is_complete": False,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }
            _atomic_write_json(manifest_path, manifest_data)

        missing_batches = [b for b in range(total_batches) if b not in completed_batches]
        if missing_batches:
            self._ensure_model_loaded()
            t_start = time.time()
            done_so_far = len(completed_batches)

            for b in missing_batches:
                b_start = b * batch_size
                b_end = min(b_start + batch_size, total_chunks)
                batch_slice = chunks[b_start:b_end]
                batch_texts = [c.text for c in batch_slice]

                t_b0 = time.time()
                batch_embs = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                batch_arr = np.asarray(batch_embs, dtype=np.float32)

                norms = np.linalg.norm(batch_arr, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                batch_arr = batch_arr / norms

                if cache_folder is not None:
                    batch_file = cache_folder / f"batch_{b:05d}.npy"
                    meta_file = cache_folder / f"batch_{b:05d}.meta.json"

                    _atomic_save_npy(batch_file, batch_arr)

                    h_batch = hashlib.sha256()
                    with open(batch_file, "rb") as bf:
                        for block in iter(lambda: bf.read(65536), b""):
                            h_batch.update(block)

                    batch_meta = {
                        "batch_index": b,
                        "start_idx": b_start,
                        "end_idx": b_end,
                        "num_chunks": len(batch_slice),
                        "chunk_ids": [c.chunk_id for c in batch_slice],
                        "sha256": h_batch.hexdigest(),
                        "shape": list(batch_arr.shape),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    }
                    _atomic_write_json(meta_file, batch_meta)

                    completed_batches.add(b)
                    manifest_data["completed_batches"] = sorted(completed_batches)
                    manifest_data["updated_at"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
                    )
                    _atomic_write_json(manifest_path, manifest_data)

                done_so_far += 1
                elapsed = time.time() - t_start
                batches_remaining = total_batches - done_so_far
                eta = (elapsed / done_so_far) * batches_remaining if done_so_far > 0 else 0.0
                ram_mb = (
                    psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
                    if psutil is not None
                    else 0.0
                )

                logger.info(
                    f"[Dense {self.device.upper()}] Batch {done_so_far}/{total_batches} "
                    f"({(done_so_far/total_batches)*100:.1f}%) | "
                    f"Batch time: {time.time()-t_b0:.2f}s | "
                    f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                    f"RAM: {ram_mb:.1f} MB"
                )

                del batch_texts, batch_embs, batch_arr

        if cache_folder is not None:
            embeddings_path = cache_folder / "embeddings.npy"
            merged_embeddings = np.zeros((total_chunks, self.embedding_dim), dtype=np.float32)

            for b in range(total_batches):
                b_start = b * batch_size
                b_end = min(b_start + batch_size, total_chunks)
                batch_file = cache_folder / f"batch_{b:05d}.npy"
                merged_embeddings[b_start:b_end] = np.load(str(batch_file))

            _atomic_save_npy(embeddings_path, merged_embeddings)

            manifest_data["is_complete"] = True
            manifest_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            _atomic_write_json(manifest_path, manifest_data)

            self.embeddings = merged_embeddings
        else:
            all_arrs = []
            self._ensure_model_loaded()
            for b in range(total_batches):
                b_start = b * batch_size
                b_end = min(b_start + batch_size, total_chunks)
                batch_slice = chunks[b_start:b_end]
                batch_texts = [c.text for c in batch_slice]
                embs = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                arr = np.asarray(embs, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                all_arrs.append(arr / norms)
            self.embeddings = np.vstack(all_arrs)

        self.is_fitted = True
        return self

    def retrieve_chunks(self, query: str, top_k: int = 50) -> list[ChunkRetrievalResult]:
        if not self.is_fitted or self.embeddings is None:
            raise RuntimeError("DenseRetriever must be fitted before retrieval.")

        clean_q = query.strip()
        if not clean_q:
            return []

        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device)

        prefixed_q = f"{self.QUERY_INSTRUCTION}{clean_q}"
        q_emb = self.model.encode([prefixed_q], normalize_embeddings=True, show_progress_bar=False)
        q_vec = np.asarray(q_emb, dtype=np.float32).flatten()

        scores = np.dot(self.embeddings, q_vec)
        num_chunks = len(self.chunks)
        k = min(top_k, num_chunks)
        if k == 0:
            return []

        if num_chunks > k:
            candidate_indices = np.argpartition(scores, -k)[-k:]
            sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-scores)

        results: list[ChunkRetrievalResult] = []
        for rank_idx, c_idx in enumerate(sorted_indices, start=1):
            score = float(scores[c_idx])
            item = self.chunks[int(c_idx)]
            snippet = item.text[:300].replace("\n", " ").strip()
            results.append(
                ChunkRetrievalResult(
                    chunk_id=item.chunk_id,
                    doc_name=item.doc_name,
                    page_index=item.page_index,
                    chunk_index=item.chunk_index,
                    score=score,
                    rank=rank_idx,
                    text_snippet=snippet,
                    char_start=item.char_start,
                    char_end=item.char_end,
                )
            )

        return results

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 100) -> list[RetrievalResult]:
        chunks = self.retrieve_chunks(query, top_k=candidate_k)
        return deduplicate_chunks_to_pages(chunks, top_k=top_k)


class HybridRRFRetriever:
    """
    Hybrid retriever fusing BM25 and Dense vector rankings using
    Reciprocal Rank Fusion (RRF):
        RRF_score(chunk) = 1 / (k + rank_bm25) + 1 / (k + rank_dense)
    where k is the fusion constant (default 60).
    """

    def __init__(
        self,
        bm25_retriever: ChunkBM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def retrieve_chunks(
        self,
        query: str,
        candidate_k: int = 100,
        top_k: int = 50,
    ) -> list[ChunkRetrievalResult]:
        bm25_res = self.bm25_retriever.retrieve_chunks(query, top_k=candidate_k)
        dense_res = self.dense_retriever.retrieve_chunks(query, top_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, ChunkRetrievalResult] = {}

        for r in bm25_res:
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (self.rrf_k + r.rank)
            chunk_map[r.chunk_id] = r

        for r in dense_res:
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (self.rrf_k + r.rank)
            if r.chunk_id not in chunk_map:
                chunk_map[r.chunk_id] = r

        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        fused_chunks: list[ChunkRetrievalResult] = []
        for rank_idx, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            src = chunk_map[cid]
            fused_chunks.append(
                ChunkRetrievalResult(
                    chunk_id=src.chunk_id,
                    doc_name=src.doc_name,
                    page_index=src.page_index,
                    chunk_index=src.chunk_index,
                    score=float(rrf_scores[cid]),
                    rank=rank_idx,
                    text_snippet=src.text_snippet,
                    char_start=src.char_start,
                    char_end=src.char_end,
                )
            )

        return fused_chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 100,
    ) -> list[RetrievalResult]:
        chunks = self.retrieve_chunks(query, candidate_k=candidate_k, top_k=candidate_k)
        return deduplicate_chunks_to_pages(chunks, top_k=top_k)


class NeuralReranker:
    """
    Cross-encoder neural reranker using BAAI/bge-reranker-base.
    Takes candidate chunks from hybrid retrieval and scores (query, chunk_text) pairs.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "cpu",
        allow_cuda: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer: Any = None
        self.model: Any = None

        if device == "cuda" and allow_cuda:
            if check_cuda_safe():
                self.device = "cuda"
            else:
                logger.warning(
                    "CUDA requested for NeuralReranker but smoke test failed. Falling back to CPU."
                )
                self.device = "cpu"
        else:
            self.device = "cpu"

    def _ensure_model_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def rerank_chunks(
        self,
        query: str,
        candidate_chunks: list[ChunkRetrievalResult],
        top_k: int | None = None,
    ) -> list[ChunkRetrievalResult]:
        if not candidate_chunks:
            return []

        self._ensure_model_loaded()
        pairs = [[query, ch.text_snippet] for ch in candidate_chunks]
        scores: list[float] = []

        import torch

        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())

        ranked_pairs = sorted(
            zip(scores, candidate_chunks),
            key=lambda x: (-x[0], x[1].chunk_id),
        )

        limit = top_k if top_k is not None else len(ranked_pairs)
        reranked_chunks: list[ChunkRetrievalResult] = []
        for rank_idx, (score, ch) in enumerate(ranked_pairs[:limit], start=1):
            reranked_chunks.append(
                ChunkRetrievalResult(
                    chunk_id=ch.chunk_id,
                    doc_name=ch.doc_name,
                    page_index=ch.page_index,
                    chunk_index=ch.chunk_index,
                    score=float(score),
                    rank=rank_idx,
                    text_snippet=ch.text_snippet,
                    char_start=ch.char_start,
                    char_end=ch.char_end,
                )
            )

        return reranked_chunks

    def rerank_to_pages(
        self,
        query: str,
        candidate_chunks: list[ChunkRetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        reranked_chunks = self.rerank_chunks(query, candidate_chunks)
        return deduplicate_chunks_to_pages(reranked_chunks, top_k=top_k)


class HybridNeuralRerankRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Hybrid RRF retrieval retrieves top candidate chunks (default 30).
    Stage 2: BGE Cross-Encoder reranks the 30 candidate chunks.
    Final: Chunks are deduplicated to unique source pages.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRRFRetriever,
        reranker: NeuralReranker,
        rerank_candidate_k: int = 30,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        candidates = self.hybrid_retriever.retrieve_chunks(
            query,
            candidate_k=self.rerank_candidate_k * 2,
            top_k=self.rerank_candidate_k,
        )
        return self.reranker.rerank_to_pages(query, candidates, top_k=top_k)

