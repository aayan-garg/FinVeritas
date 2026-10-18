#!/usr/bin/env python3
"""
Test DenseRetriever CPU execution, incremental caching, and resume behavior
on 256 real FinanceBench chunks.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from finveritas.rag_indexing import (  # noqa: E402
    DenseRetriever,
    build_chunk_corpus,
    load_page_corpus,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("test_cpu_256")


def main() -> int:
    corpus_path = Path("data/interim/financebench/page_corpus.jsonl")
    if not corpus_path.is_file():
        logger.error(f"Missing page corpus at {corpus_path}")
        return 1

    # 1. Load pages and chunk
    logger.info("Loading page corpus...")
    pages = load_page_corpus(corpus_path)
    logger.info(f"Loaded {len(pages)} pages. Building overlapping chunks (256/64)...")
    all_chunks, _ = build_chunk_corpus(pages, chunk_size=256, chunk_overlap=64)

    # 2. Select at most 256 real chunks
    sample_chunks = all_chunks[:256]
    logger.info(f"Selected {len(sample_chunks)} real chunks for CPU staged test.")
    logger.info(
        f"Chunk 0: ID={sample_chunks[0].chunk_id}, doc={sample_chunks[0].doc_name}, "
        f"page={sample_chunks[0].page_index}"
    )
    logger.info(
        f"Chunk 255: ID={sample_chunks[-1].chunk_id}, doc={sample_chunks[-1].doc_name}, "
        f"page={sample_chunks[-1].page_index}"
    )

    test_cache_dir = Path("artifacts/advanced_retrieval/cache_test_256")
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)
    test_cache_dir.mkdir(parents=True, exist_ok=True)

    # 3. First run: encode 256 chunks on CPU (batch_size=32 -> 8 batches)
    logger.info("\n--- Phase 3.1: Fresh CPU Encoding (8 batches of 32 chunks) ---")
    retriever_1 = DenseRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=32,
        device="cpu",
        cache_dir=test_cache_dir,
        chunking_config={"chunk_size": 256, "chunk_overlap": 64},
    )

    t0 = time.time()
    retriever_1.fit(sample_chunks)
    elapsed_1 = time.time() - t0
    logger.info(f"Fresh encoding completed in {elapsed_1:.2f}s.")

    # 4. Verify embedding shapes and unit norms
    assert retriever_1.embeddings is not None
    assert retriever_1.embeddings.shape == (256, 384), (
        f"Wrong shape: {retriever_1.embeddings.shape}"
    )
    norms = np.linalg.norm(retriever_1.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not unit normalized!"
    logger.info(f"Shape verified: {retriever_1.embeddings.shape}")
    logger.info(
        f"Unit norms verified: min={norms.min():.4f}, max={norms.max():.4f}, "
        f"mean={norms.mean():.4f}"
    )

    # Verify manifest
    cache_folder = retriever_1._get_cache_folder(sample_chunks)
    assert cache_folder is not None
    manifest_path = cache_folder / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    done_batches = len(manifest.get("completed_batches", []))
    total_b = manifest.get("total_batches")
    logger.info(
        f"Manifest: is_complete={manifest.get('is_complete')}, "
        f"batches={done_batches}/{total_b}"
    )
    assert manifest.get("is_complete") is True
    assert manifest.get("num_chunks") == 256
    assert len(manifest.get("completed_batches", [])) == 8

    # 5. Second run: test instant resume without re-encoding
    logger.info("\n--- Phase 3.2: Cache Reuse & Instant Resume Test ---")
    retriever_2 = DenseRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=32,
        device="cpu",
        cache_dir=test_cache_dir,
    )

    t0_resume = time.time()
    retriever_2.fit(sample_chunks)
    elapsed_resume = time.time() - t0_resume
    logger.info(f"Resumed from cache in {elapsed_resume:.4f}s.")
    assert retriever_2.embeddings is not None
    assert np.allclose(retriever_1.embeddings, retriever_2.embeddings)
    assert retriever_2.model is None, "Model was loaded even though embeddings were cached!"
    logger.info("Resume test passed: Model was NOT loaded, zero batches recomputed.")

    # 6. Test retrieval query across the 256 chunks
    logger.info("\n--- Phase 3.3: Dense Query Retrieval Test on 256 Chunks ---")
    query = "What were capital expenditures and cash flow?"
    top_chunks = retriever_1.retrieve_chunks(query, top_k=3)
    logger.info(f"Query: '{query}'")
    for r in top_chunks:
        logger.info(
            f"  Rank {r.rank}: Doc={r.doc_name}, Page={r.page_index}, "
            f"Score={r.score:.4f}, Snippet={r.text_snippet[:60]}..."
        )

    top_pages = retriever_1.retrieve(query, top_k=3)
    logger.info("Deduplicated Top Pages:")
    for p in top_pages:
        logger.info(
            f"  Rank {p.rank}: Doc={p.doc_name}, Page={p.page_index}, Score={p.score:.4f}"
        )

    # Clean up test cache directory to avoid leaving non-tracked artifacts
    shutil.rmtree(test_cache_dir)
    logger.info("\nCleaned up test cache directory.")

    print("\n[SUCCESS] ALL 256-CHUNK CPU STAGED TESTS PASSED PERFECTLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
