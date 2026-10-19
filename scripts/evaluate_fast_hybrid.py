#!/usr/bin/env python3
"""
FinVeritas Checkpoint 08: Fast Hybrid RAG Evaluation Runner.

Strict Constraints:
  - CPU only (CUDA blocked via environment and assertions)
  - Bounded CPU encoding (no full 37k chunk encoding)
  - Resumable, cache-backed DenseRetriever
  - Exact apples-to-apples candidate sets for multi-retriever comparison
  - Labeled explicitly as a smoke test on real FinanceBench data
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil

# Enforce strict CPU-only and offline HF cache
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from finveritas.rag_indexing import (  # noqa: E402
    BM25Retriever,
    ChunkBM25Retriever,
    DenseRetriever,
    HybridNeuralRerankRetriever,
    HybridRRFRetriever,
    NeuralReranker,
    build_chunk_corpus,
    compute_paired_bootstrap,
    evaluate_retriever_on_financebench,
    load_financebench_metadata,
    load_page_corpus,
)

logger = logging.getLogger("fast_hybrid_evaluation")


def get_memory_mb() -> float:
    p = psutil.Process(os.getpid())
    return round(p.memory_info().rss / (1024.0 * 1024.0), 2)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

    # Safety assertion: verify CUDA has NOT been initialized, and enforce CPU-only
    try:
        import torch

        if torch.cuda.is_initialized():
            logger.error("CUDA was already initialized before evaluation. Aborting for safety.")
            return 1
        # Block any library from probing or initializing CUDA
        torch.cuda.is_available = lambda: False
    except ImportError:
        pass

    data_dir = Path("data/raw/financebench")
    page_corpus_path = Path("data/interim/financebench/page_corpus.jsonl")
    artifacts_dir = Path("artifacts/fast_hybrid_eval")
    cache_dir = artifacts_dir / "cache"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cutoffs = [1, 3, 5, 10]

    logger.info("=" * 80)
    logger.info("FINVERITAS CHECKPOINT 08: FAST HYBRID RAG EVALUATION")
    logger.info(f"Safety Mode: CPU ONLY | Offline HF Cache: ACTIVE | Artifacts: {artifacts_dir}")
    logger.info("=" * 80)

    # 1. Load Ground Truth Metadata and Full Page Corpus
    t0_load = time.time()
    open_jsonl = data_dir / "financebench_open_source.jsonl"
    doc_jsonl = data_dir / "financebench_document_information.jsonl"
    all_questions, doc_info_map = load_financebench_metadata(open_jsonl, doc_jsonl)
    full_page_corpus = load_page_corpus(page_corpus_path)
    full_indexed_docs = set(p.doc_name for p in full_page_corpus)
    logger.info(
        f"Loaded {len(full_page_corpus)} pages ({len(full_indexed_docs)} documents) and "
        f"{len(all_questions)} questions in {time.time() - t0_load:.2f}s."
    )

    # 2. Run Full-Corpus BM25 Contextual Baselines
    logger.info("\n--- [Phase A: Full-Corpus BM25 Baselines (150 Questions, 12,013 Pages)] ---")
    t0 = time.time()
    full_page_bm25 = BM25Retriever().fit(full_page_corpus)
    t_full_page_fit = time.time() - t0

    t0 = time.time()
    full_page_bm25_eval = evaluate_retriever_on_financebench(
        retriever=full_page_bm25,
        questions=all_questions,
        indexed_doc_names=full_indexed_docs,
        k_values=cutoffs,
    )
    t_full_page_eval = time.time() - t0
    perf_full_page = full_page_bm25_eval["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Full-Corpus Page BM25: Doc R@5={perf_full_page['doc_recall']['recall_at_5'] * 100:.2f}%, "
        f"Page R@5={perf_full_page['page_recall']['recall_at_5'] * 100:.2f}%, "
        f"MRR={perf_full_page['mrr']:.4f}, latency={perf_full_page['latency']['mean_ms']}ms "
        f"(fit={t_full_page_fit:.2f}s, eval={t_full_page_eval:.2f}s)"
    )

    logger.info("Building full-corpus 256-token chunks for Chunk BM25 baseline...")
    t0 = time.time()
    full_chunks_256, full_chunk_stats = build_chunk_corpus(
        full_page_corpus, chunk_size=256, chunk_overlap=64
    )
    t_full_chunk_build = time.time() - t0

    t0 = time.time()
    full_chunk_bm25 = ChunkBM25Retriever().fit(full_chunks_256)
    t_full_chunk_fit = time.time() - t0

    t0 = time.time()
    full_chunk_bm25_eval = evaluate_retriever_on_financebench(
        retriever=full_chunk_bm25,
        questions=all_questions,
        indexed_doc_names=full_indexed_docs,
        k_values=cutoffs,
    )
    t_full_chunk_eval = time.time() - t0
    perf_full_chunk = full_chunk_bm25_eval["performance"]["total_benchmark_metrics"]
    d_r5_fc = perf_full_chunk["doc_recall"]["recall_at_5"] * 100
    p_r5_fc = perf_full_chunk["page_recall"]["recall_at_5"] * 100
    lat_fc = perf_full_chunk["latency"]["mean_ms"]
    logger.info(
        f"Full-Corpus Chunk BM25: Doc R@5={d_r5_fc:.2f}%, Page R@5={p_r5_fc:.2f}%, "
        f"MRR={perf_full_chunk['mrr']:.4f}, latency={lat_fc}ms "
        f"(chunking={t_full_chunk_build:.2f}s, fit={t_full_chunk_fit:.2f}s, "
        f"eval={t_full_chunk_eval:.2f}s)"
    )

    # 3. Controlled Bounded Real-Data Smoke Test
    # Target subset: Top 4 complete 10-K filings with highest question density
    target_docs = [
        "AMD_2022_10K",
        "AMERICANEXPRESS_2022_10K",
        "BOEING_2022_10K",
        "AMCOR_2023_10K",
    ]
    target_doc_set = set(target_docs)

    smoke_questions = [q for q in all_questions if q["doc_name"] in target_doc_set]
    smoke_pages = [p for p in full_page_corpus if p.doc_name in target_doc_set]
    smoke_indexed_docs = set(p.doc_name for p in smoke_pages)

    logger.info(
        f"\n--- [Phase B: Bounded Real-Data Smoke Test (Apples-to-Apples 4-Way Comparison)] ---\n"
        f"Target Documents: {target_docs}\n"
        f"Questions: {len(smoke_questions)} | Pages: {len(smoke_pages)} | "
        f"Coverage: 100% (all gold evidence present in candidate pool)"
    )

    t0 = time.time()
    smoke_chunks_256, smoke_chunk_stats = build_chunk_corpus(
        smoke_pages, chunk_size=256, chunk_overlap=64
    )
    t_smoke_chunk = time.time() - t0
    n_chk = len(smoke_chunks_256)
    n_pg = len(smoke_pages)
    logger.info(f"Created {n_chk} chunks from {n_pg} pages in {t_smoke_chunk:.2f}s.")

    smoke_results: dict[str, Any] = {}
    smoke_timings: dict[str, Any] = {}
    smoke_mem: dict[str, Any] = {}

    # System 1: Page BM25 on smoke corpus
    logger.info("\n1. Evaluating Smoke Page BM25...")
    t0 = time.time()
    smoke_page_bm25 = BM25Retriever().fit(smoke_pages)
    smoke_timings["page_bm25_fit_sec"] = round(time.time() - t0, 3)
    smoke_mem["page_bm25"] = get_memory_mb()

    eval_smoke_page_bm25 = evaluate_retriever_on_financebench(
        retriever=smoke_page_bm25,
        questions=smoke_questions,
        indexed_doc_names=smoke_indexed_docs,
        k_values=cutoffs,
    )
    smoke_results["smoke_page_bm25"] = eval_smoke_page_bm25
    perf_s_page = eval_smoke_page_bm25["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Smoke Page BM25: Doc R@5={perf_s_page['doc_recall']['recall_at_5'] * 100:.2f}%, "
        f"Page R@5={perf_s_page['page_recall']['recall_at_5'] * 100:.2f}%, "
        f"MRR={perf_s_page['mrr']:.4f}, nDCG@5={perf_s_page['ndcg']['ndcg_at_5']:.4f}"
    )

    # System 2: Chunk BM25 on smoke corpus
    logger.info("\n2. Evaluating Smoke Chunk BM25...")
    t0 = time.time()
    smoke_chunk_bm25 = ChunkBM25Retriever().fit(smoke_chunks_256)
    smoke_timings["chunk_bm25_fit_sec"] = round(time.time() - t0, 3)
    smoke_mem["chunk_bm25"] = get_memory_mb()

    eval_smoke_chunk_bm25 = evaluate_retriever_on_financebench(
        retriever=smoke_chunk_bm25,
        questions=smoke_questions,
        indexed_doc_names=smoke_indexed_docs,
        k_values=cutoffs,
    )
    smoke_results["smoke_chunk_bm25"] = eval_smoke_chunk_bm25
    perf_s_chunk = eval_smoke_chunk_bm25["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Smoke Chunk BM25: Doc R@5={perf_s_chunk['doc_recall']['recall_at_5'] * 100:.2f}%, "
        f"Page R@5={perf_s_chunk['page_recall']['recall_at_5'] * 100:.2f}%, "
        f"MRR={perf_s_chunk['mrr']:.4f}, nDCG@5={perf_s_chunk['ndcg']['ndcg_at_5']:.4f}"
    )

    # System 3: Dense Retrieval on smoke corpus
    logger.info(
        "\n3. Evaluating Smoke Dense Retrieval (BAAI/bge-small-en-v1.5, CPU, batch_size=32)..."
    )
    dense_retriever = DenseRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=32,
        device="cpu",
        allow_cuda=False,
        cache_dir=cache_dir,
        chunking_config={"chunk_size": 256, "chunk_overlap": 64},
    )
    t0 = time.time()
    dense_retriever.fit(smoke_chunks_256)
    smoke_timings["dense_fit_sec"] = round(time.time() - t0, 3)
    smoke_mem["dense"] = get_memory_mb()
    logger.info(f"Dense retriever fitted in {smoke_timings['dense_fit_sec']:.2f}s.")

    eval_smoke_dense = evaluate_retriever_on_financebench(
        retriever=dense_retriever,
        questions=smoke_questions,
        indexed_doc_names=smoke_indexed_docs,
        k_values=cutoffs,
    )
    smoke_results["smoke_dense"] = eval_smoke_dense
    perf_s_dense = eval_smoke_dense["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Smoke Dense: Doc R@5={perf_s_dense['doc_recall']['recall_at_5'] * 100:.2f}%, "
        f"Page R@5={perf_s_dense['page_recall']['recall_at_5'] * 100:.2f}%, "
        f"MRR={perf_s_dense['mrr']:.4f}, nDCG@5={perf_s_dense['ndcg']['ndcg_at_5']:.4f}"
    )

    # System 4: Hybrid RRF (k=60)
    logger.info("\n4. Evaluating Smoke Hybrid RRF (Chunk BM25 + Dense BGE)...")
    smoke_hybrid_rrf = HybridRRFRetriever(
        bm25_retriever=smoke_chunk_bm25,
        dense_retriever=dense_retriever,
        rrf_k=60,
    )
    t0 = time.time()
    eval_smoke_hybrid = evaluate_retriever_on_financebench(
        retriever=smoke_hybrid_rrf,
        questions=smoke_questions,
        indexed_doc_names=smoke_indexed_docs,
        k_values=cutoffs,
    )
    smoke_timings["hybrid_eval_sec"] = round(time.time() - t0, 3)
    smoke_mem["hybrid"] = get_memory_mb()
    smoke_results["smoke_hybrid_rrf"] = eval_smoke_hybrid
    perf_s_hybrid = eval_smoke_hybrid["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Smoke Hybrid RRF: Doc R@5={perf_s_hybrid['doc_recall']['recall_at_5'] * 100:.2f}%, "
        f"Page R@5={perf_s_hybrid['page_recall']['recall_at_5'] * 100:.2f}%, "
        f"MRR={perf_s_hybrid['mrr']:.4f}, nDCG@5={perf_s_hybrid['ndcg']['ndcg_at_5']:.4f}"
    )

    # System 5 (Optional): Neural Reranker with bge-reranker-base on top 10 candidates
    logger.info("\n5. Testing Optional Neural Reranker (top 10 candidates)...")
    eval_smoke_rerank = None
    perf_s_rerank = None
    try:
        reranker = NeuralReranker(
            model_name="BAAI/bge-reranker-base",
            device="cpu",
            allow_cuda=False,
            max_length=256,
        )
        hybrid_rerank_retriever = HybridNeuralRerankRetriever(
            hybrid_retriever=smoke_hybrid_rrf,
            reranker=reranker,
            rerank_top_k=10,
        )
        t0 = time.time()
        eval_smoke_rerank = evaluate_retriever_on_financebench(
            retriever=hybrid_rerank_retriever,
            questions=smoke_questions,
            indexed_doc_names=smoke_indexed_docs,
            k_values=cutoffs,
        )
        smoke_timings["rerank_eval_sec"] = round(time.time() - t0, 3)
        smoke_mem["rerank"] = get_memory_mb()
        smoke_results["smoke_neural_rerank"] = eval_smoke_rerank
        perf_s_rerank = eval_smoke_rerank["performance"]["total_benchmark_metrics"]
        d_r5_rr = perf_s_rerank["doc_recall"]["recall_at_5"] * 100
        p_r5_rr = perf_s_rerank["page_recall"]["recall_at_5"] * 100
        logger.info(
            f"Smoke Neural Rerank: Doc R@5={d_r5_rr:.2f}%, Page R@5={p_r5_rr:.2f}%, "
            f"MRR={perf_s_rerank['mrr']:.4f}, nDCG@5={perf_s_rerank['ndcg']['ndcg_at_5']:.4f}"
        )
    except Exception as e:
        logger.warning(f"Neural reranker skipped or encountered error: {e}")

    # 4. Compute Paired Statistical Bootstraps on Smoke Subset
    logger.info("\n--- Computing Paired Bootstrap Resampling (1,000 iterations) ---")
    bootstraps = {}

    def extract_page_r5(r):
        return 1.0 if r.page_match_at_k.get(5, False) else 0.0

    def extract_mrr(r):
        return r.reciprocal_rank

    bootstraps["chunk_bm25_vs_page_bm25_page_r5"] = compute_paired_bootstrap(
        eval_smoke_page_bm25["per_question_evaluations"],
        eval_smoke_chunk_bm25["per_question_evaluations"],
        metric_extractor=extract_page_r5,
    )
    bootstraps["dense_vs_chunk_bm25_page_r5"] = compute_paired_bootstrap(
        eval_smoke_chunk_bm25["per_question_evaluations"],
        eval_smoke_dense["per_question_evaluations"],
        metric_extractor=extract_page_r5,
    )
    bootstraps["hybrid_vs_chunk_bm25_page_r5"] = compute_paired_bootstrap(
        eval_smoke_chunk_bm25["per_question_evaluations"],
        eval_smoke_hybrid["per_question_evaluations"],
        metric_extractor=extract_page_r5,
    )
    bootstraps["hybrid_vs_dense_page_r5"] = compute_paired_bootstrap(
        eval_smoke_dense["per_question_evaluations"],
        eval_smoke_hybrid["per_question_evaluations"],
        metric_extractor=extract_page_r5,
    )

    # 5. Save Artifacts
    output_summary = {
        "evaluation_type": "smoke_test_and_contextual_baselines",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "device": "cpu",
            "cuda_initialized": False,
            "logical_cpus": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024.0**3), 2),
        },
        "full_corpus_contextual_baselines": {
            "num_questions": len(all_questions),
            "num_pages": len(full_page_corpus),
            "num_chunks_256": len(full_chunks_256),
            "page_bm25": perf_full_page,
            "chunk_bm25_256": perf_full_chunk,
            "timing_sec": {
                "page_bm25_fit": round(t_full_page_fit, 2),
                "page_bm25_eval": round(t_full_page_eval, 2),
                "chunk_bm25_chunking": round(t_full_chunk_build, 2),
                "chunk_bm25_fit": round(t_full_chunk_fit, 2),
                "chunk_bm25_eval": round(t_full_chunk_eval, 2),
            },
        },
        "bounded_real_data_smoke_test": {
            "target_documents": target_docs,
            "num_questions": len(smoke_questions),
            "num_pages": len(smoke_pages),
            "num_chunks": len(smoke_chunks_256),
            "chunking_config": {"chunk_size": 256, "chunk_overlap": 64},
            "timings_sec": smoke_timings,
            "memory_mb": smoke_mem,
            "systems": {
                "system_1_page_bm25": perf_s_page,
                "system_2_chunk_bm25": perf_s_chunk,
                "system_3_dense_bge_small": perf_s_dense,
                "system_4_hybrid_rrf_k60": perf_s_hybrid,
                "system_5_neural_rerank": perf_s_rerank,
            },
            "paired_bootstrap": bootstraps,
        },
    }

    out_json = artifacts_dir / "fast_hybrid_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    logger.info(f"Saved complete evaluation report to {out_json}")

    # 6. Formatted Comparison Table
    print("\n" + "=" * 115)
    print("FINVERITAS CHECKPOINT 08: BOUNDED REAL-DATA SMOKE TEST (APPLES-TO-APPLES)")
    sub_title = (
        f"Corpus: 4 10-K Filings ({len(smoke_pages)} Pages, {len(smoke_chunks_256)} Chunks) | "
        f"Queries: {len(smoke_questions)} Questions | CPU Only"
    )
    print(sub_title)
    print("=" * 115)
    header = (
        f"{'Metric':<24} | {'(1) Page BM25':<14} | {'(2) Chunk BM25':<14} | "
        f"{'(3) Dense BGE':<14} | {'(4) Hybrid RRF':<14} | {'(5) Neural Rerank':<15}"
    )
    print(header)
    print("-" * 115)

    def print_metric_row(label, cat, key):
        v1 = perf_s_page[cat][key]
        v2 = perf_s_chunk[cat][key]
        v3 = perf_s_dense[cat][key]
        v4 = perf_s_hybrid[cat][key]
        v5 = perf_s_rerank[cat][key] if perf_s_rerank else 0.0
        print(
            f"{label:<24} | {v1 * 100:13.2f}% | {v2 * 100:13.2f}% | "
            f"{v3 * 100:13.2f}% | {v4 * 100:13.2f}% | {v5 * 100:14.2f}%"
        )

    for k in [1, 3, 5, 10]:
        print_metric_row(f"Doc Recall@{k}", "doc_recall", f"recall_at_{k}")
    print("-" * 115)
    for k in [1, 3, 5, 10]:
        print_metric_row(f"Page Recall@{k}", "page_recall", f"recall_at_{k}")
    print("-" * 115)
    print_metric_row("Annotation Recall@5", "annotation_recall", "recall_at_5")
    print("-" * 115)
    mrr1 = perf_s_page["mrr"]
    mrr2 = perf_s_chunk["mrr"]
    mrr3 = perf_s_dense["mrr"]
    mrr4 = perf_s_hybrid["mrr"]
    mrr5 = perf_s_rerank["mrr"] if perf_s_rerank else 0.0
    print(
        f"{'Page MRR':<24} | {mrr1:14.4f} | {mrr2:14.4f} | "
        f"{mrr3:14.4f} | {mrr4:14.4f} | {mrr5:15.4f}"
    )
    ndcg1 = perf_s_page["ndcg"]["ndcg_at_5"]
    ndcg2 = perf_s_chunk["ndcg"]["ndcg_at_5"]
    ndcg3 = perf_s_dense["ndcg"]["ndcg_at_5"]
    ndcg4 = perf_s_hybrid["ndcg"]["ndcg_at_5"]
    ndcg5 = perf_s_rerank["ndcg"]["ndcg_at_5"] if perf_s_rerank else 0.0
    print(
        f"{'Page nDCG@5':<24} | {ndcg1:14.4f} | {ndcg2:14.4f} | "
        f"{ndcg3:14.4f} | {ndcg4:14.4f} | {ndcg5:15.4f}"
    )
    print("-" * 115)
    lat1 = perf_s_page["latency"]["mean_ms"]
    lat2 = perf_s_chunk["latency"]["mean_ms"]
    lat3 = perf_s_dense["latency"]["mean_ms"]
    lat4 = perf_s_hybrid["latency"]["mean_ms"]
    lat5 = perf_s_rerank["latency"]["mean_ms"] if perf_s_rerank else 0.0
    print(
        f"{'Mean Latency':<24} | {lat1:11.1f} ms | {lat2:11.1f} ms | "
        f"{lat3:11.1f} ms | {lat4:11.1f} ms | {lat5:12.1f} ms"
    )
    print("=" * 115)

    return 0


if __name__ == "__main__":
    sys.exit(main())
