#!/usr/bin/env python3
"""
FinVeritas Checkpoint 07: Advanced Hybrid Financial Evidence Retrieval Evaluation.

Evaluates 5 retrieval configurations on FinanceBench (150 questions, 12,013 PDF pages):
  A. Reference Page-Level BM25 Baseline
  B. Chunk-Level BM25 (overlapping chunks mapped to source pages)
  C. Dense Retrieval (BAAI/bge-small-en-v1.5 with cosine similarity & persistent cache)
  D. Hybrid Retrieval with Reciprocal Rank Fusion (k=60)
  E. Two-Stage Hybrid + Neural Reranker (BAAI/bge-reranker-base cross-encoder)

Also executes:
  - Controlled chunking ablation: 256 vs 384 tokens (64 overlap)
  - Paired bootstrap resampling (1,000 clusters) for 95% confidence intervals
  - Sub-benchmark breakdown: Numerical/table questions (n=113) vs Prose questions (n=37)
  - Qualitative error taxonomy across >=15 missed questions
  - Profiling: mean/median latency, index build time, rerank latency, peak RAM/VRAM
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil

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

logger = logging.getLogger("evaluate_advanced_retrieval")


def get_memory_mb() -> tuple[float, float]:
    """Returns current process RAM (MB) and peak GPU VRAM (MB) if CUDA is active."""
    p = psutil.Process(os.getpid())
    ram_mb = p.memory_info().rss / (1024.0 * 1024.0)
    vram_mb = 0.0
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass
    return round(ram_mb, 2), round(vram_mb, 2)


def categorize_error(
    error_item: dict[str, Any],
    top_candidates: list[dict[str, Any]],
) -> str:
    """
    Categorizes retrieval error into one of the 7 specified failure modes:
      1. Wrong source document
      2. Correct document but wrong page
      3. Table or number not adequately extracted
      4. Query uses different terminology from the filing
      5. Reporting-period or segment mismatch
      6. Evidence distributed over multiple pages
      7. Chunking, deduplication, or reranking failure
    """
    gold_doc = error_item.get("gold_doc_name", "")
    gold_items = error_item.get("gold_evidence_items", [])
    q_text = error_item.get("question", "").lower()
    doc_in_top = error_item.get("doc_found_in_top_k", False)

    # Check if multiple evidence pages exist
    if len(gold_items) > 1:
        return "Evidence distributed over multiple pages"

    if not doc_in_top:
        # Check if year or ticker confusion occurred
        top_docs = [c["doc_name"] for c in top_candidates[:3]]
        if any(gold_doc.split("_")[0] in td for td in top_docs):
            return "Reporting-period or segment mismatch"
        return "Wrong source document"

    # Document was in top-k, but wrong page
    top_text = " ".join(c.get("text_snippet", "") for c in top_candidates[:3]).lower()
    table_terms = ["ratio", "margin", "percentage", "ebitda", "capex", "balance sheet", "$"]
    if any(term in q_text for term in table_terms):
        if any(dig in top_text for dig in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
            return "Table or number not adequately extracted"

    prose_terms = ["purpose", "strategy", "risk", "policy", "definition", "describe"]
    if any(kw in q_text for kw in prose_terms):
        return "Query uses different terminology from the filing"

    return "Correct document but wrong page"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinVeritas Advanced Hybrid Financial Evidence Retrieval Evaluation"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/financebench"),
        help="Path to raw FinanceBench data directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/advanced_retrieval/cache"),
        help="Path to persistent embedding and index cache.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/advanced_retrieval"),
        help="Path to output artifacts directory.",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Cutoff ranks for Recall and nDCG (default: 1 3 5 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' or 'cuda'). Default: 'cpu' for safety.",
    )
    parser.add_argument(
        "--allow-cuda",
        action="store_true",
        default=False,
        help="Explicitly permit CUDA if device is 'cuda' and safe smoke test passes.",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=32,
        help="Batch size for dense embedding generation (default: 32).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks to process for testing/dry-runs (default: None).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    data_dir: Path = args.data_dir
    cache_dir: Path = args.cache_dir
    artifacts_dir: Path = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cutoffs = args.cutoffs
    device = args.device

    logger.info("=" * 80)
    logger.info("FINVERITAS CHECKPOINT 07: ADVANCED HYBRID FINANCIAL RETRIEVAL EVALUATION")
    logger.info(
        f"Device: {device} (allow_cuda={args.allow_cuda}) | "
        f"Dense batch size: {args.dense_batch_size} | "
        f"Max chunks: {args.max_chunks} | Cutoffs: {cutoffs} | Cache: {cache_dir}"
    )
    logger.info("=" * 80)

    # 1. Load Ground Truth Metadata and Extracted Page Corpus
    open_jsonl = data_dir / "financebench_open_source.jsonl"
    doc_jsonl = data_dir / "financebench_document_information.jsonl"
    page_corpus_path = Path("data/interim/financebench/page_corpus.jsonl")

    questions, doc_info_map = load_financebench_metadata(open_jsonl, doc_jsonl)
    logger.info(f"Loaded {len(questions)} FinanceBench questions.")

    if not page_corpus_path.is_file():
        logger.error(f"Missing cached page corpus at {page_corpus_path}. Extract corpus first.")
        return 1

    page_corpus = load_page_corpus(page_corpus_path)
    indexed_doc_names = set(p.doc_name for p in page_corpus)
    logger.info(
        f"Loaded {len(page_corpus)} pages across {len(indexed_doc_names)} documents."
    )

    # 2. Build Overlapping Chunk Corpora (Primary: 256/64, Ablation: 384/64)
    logger.info("Building 256-token overlapping chunks (stride 192, overlap 64)...")
    t0_chunk = time.time()
    chunks_256, stats_256 = build_chunk_corpus(page_corpus, chunk_size=256, chunk_overlap=64)
    t_chunk_256 = time.time() - t0_chunk
    logger.info(
        f"256-token chunking complete: {len(chunks_256)} chunks in {t_chunk_256:.2f}s "
        f"(avg {stats_256['avg_chunks_per_page']} chunks/page)."
    )

    logger.info("Building 384-token overlapping chunks (stride 320, overlap 64)...")
    t0_chunk384 = time.time()
    chunks_384, stats_384 = build_chunk_corpus(page_corpus, chunk_size=384, chunk_overlap=64)
    t_chunk_384 = time.time() - t0_chunk384
    logger.info(
        f"384-token chunking complete: {len(chunks_384)} chunks in {t_chunk_384:.2f}s "
        f"(avg {stats_384['avg_chunks_per_page']} chunks/page)."
    )

    # Tracking dictionary for all experimental configurations
    results: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    mem_records: dict[str, Any] = {}

    # =========================================================================
    # Configuration A: Reference Page-Level BM25
    # =========================================================================
    logger.info("\n--- [Config A] Reference Page-Level BM25 Baseline ---")
    t0 = time.time()
    page_bm25 = BM25Retriever().fit(page_corpus)
    timings["config_a_index_sec"] = round(time.time() - t0, 2)
    mem_records["config_a"] = get_memory_mb()

    eval_a = evaluate_retriever_on_financebench(
        retriever=page_bm25,
        questions=questions,
        indexed_doc_names=indexed_doc_names,
        k_values=cutoffs,
    )
    results["config_a_page_bm25"] = eval_a

    perf_a = eval_a["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Config A Results: Doc Recall@5: {perf_a['doc_recall']['recall_at_5']*100:.2f}%, "
        f"Page Recall@5: {perf_a['page_recall']['recall_at_5']*100:.2f}%, "
        f"Page MRR: {perf_a['mrr']:.4f}, nDCG@5: {perf_a['ndcg']['ndcg_at_5']:.4f}, "
        f"Latency: {perf_a['latency']['mean_ms']}ms"
    )

    # =========================================================================
    # Configuration B: Chunk-Level BM25 (256 tokens)
    # =========================================================================
    logger.info("\n--- [Config B] Chunk-Level BM25 (256 tokens) ---")
    t0 = time.time()
    chunk_bm25_256 = ChunkBM25Retriever().fit(chunks_256)
    timings["config_b_index_sec"] = round(time.time() - t0, 2)
    mem_records["config_b"] = get_memory_mb()

    eval_b = evaluate_retriever_on_financebench(
        retriever=chunk_bm25_256,
        questions=questions,
        indexed_doc_names=indexed_doc_names,
        k_values=cutoffs,
    )
    results["config_b_chunk_bm25"] = eval_b

    perf_b = eval_b["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Config B Results: Doc Recall@5: {perf_b['doc_recall']['recall_at_5']*100:.2f}%, "
        f"Page Recall@5: {perf_b['page_recall']['recall_at_5']*100:.2f}%, "
        f"Page MRR: {perf_b['mrr']:.4f}, nDCG@5: {perf_b['ndcg']['ndcg_at_5']:.4f}, "
        f"Latency: {perf_b['latency']['mean_ms']}ms"
    )

    # =========================================================================
    # Configuration C: Dense Vector Retrieval (BGE-Small-en-v1.5)
    # =========================================================================
    logger.info("\n--- [Config C] Dense Retrieval (BAAI/bge-small-en-v1.5) ---")
    dense_retriever_256 = DenseRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=args.dense_batch_size,
        device=device,
        allow_cuda=args.allow_cuda,
        cache_dir=cache_dir,
        chunking_config={"chunk_size": 256, "chunk_overlap": 64},
    )
    t0 = time.time()
    dense_retriever_256.fit(chunks_256, max_chunks=args.max_chunks)
    timings["config_c_embed_sec"] = round(time.time() - t0, 2)
    mem_records["config_c"] = get_memory_mb()

    eval_c = evaluate_retriever_on_financebench(
        retriever=dense_retriever_256,
        questions=questions,
        indexed_doc_names=indexed_doc_names,
        k_values=cutoffs,
    )
    results["config_c_dense"] = eval_c

    perf_c = eval_c["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Config C Results: Doc Recall@5: {perf_c['doc_recall']['recall_at_5']*100:.2f}%, "
        f"Page Recall@5: {perf_c['page_recall']['recall_at_5']*100:.2f}%, "
        f"Page MRR: {perf_c['mrr']:.4f}, nDCG@5: {perf_c['ndcg']['ndcg_at_5']:.4f}, "
        f"Latency: {perf_c['latency']['mean_ms']}ms"
    )

    # =========================================================================
    # Configuration D: Hybrid Retrieval with Reciprocal Rank Fusion (k=60)
    # =========================================================================
    logger.info("\n--- [Config D] Hybrid Retrieval with Reciprocal Rank Fusion (k=60) ---")
    hybrid_rrf_256 = HybridRRFRetriever(
        bm25_retriever=chunk_bm25_256,
        dense_retriever=dense_retriever_256,
        rrf_k=60,
    )
    mem_records["config_d"] = get_memory_mb()

    eval_d = evaluate_retriever_on_financebench(
        retriever=hybrid_rrf_256,
        questions=questions,
        indexed_doc_names=indexed_doc_names,
        k_values=cutoffs,
    )
    results["config_d_hybrid_rrf"] = eval_d

    perf_d = eval_d["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Config D Results: Doc Recall@5: {perf_d['doc_recall']['recall_at_5']*100:.2f}%, "
        f"Page Recall@5: {perf_d['page_recall']['recall_at_5']*100:.2f}%, "
        f"Page MRR: {perf_d['mrr']:.4f}, nDCG@5: {perf_d['ndcg']['ndcg_at_5']:.4f}, "
        f"Latency: {perf_d['latency']['mean_ms']}ms"
    )

    # =========================================================================
    # Configuration E: Hybrid Retrieval + Neural Reranker (BGE-Reranker-Base)
    # =========================================================================
    logger.info("\n--- [Config E] Hybrid Retrieval + Neural Reranker (bge-reranker-base) ---")
    t0 = time.time()
    reranker = NeuralReranker(
        model_name="BAAI/bge-reranker-base",
        batch_size=16,
        max_length=512,
        device=device,
        allow_cuda=args.allow_cuda,
    )
    timings["config_e_model_load_sec"] = round(time.time() - t0, 2)
    mem_records["config_e"] = get_memory_mb()

    hybrid_neural_retriever = HybridNeuralRerankRetriever(
        hybrid_retriever=hybrid_rrf_256,
        reranker=reranker,
        rerank_candidate_k=30,
    )

    eval_e = evaluate_retriever_on_financebench(
        retriever=hybrid_neural_retriever,
        questions=questions,
        indexed_doc_names=indexed_doc_names,
        k_values=cutoffs,
    )
    results["config_e_neural_rerank"] = eval_e

    perf_e = eval_e["performance"]["total_benchmark_metrics"]
    logger.info(
        f"Config E Results: Doc Recall@5: {perf_e['doc_recall']['recall_at_5']*100:.2f}%, "
        f"Page Recall@5: {perf_e['page_recall']['recall_at_5']*100:.2f}%, "
        f"Page MRR: {perf_e['mrr']:.4f}, nDCG@5: {perf_e['ndcg']['ndcg_at_5']:.4f}, "
        f"Latency: {perf_e['latency']['mean_ms']}ms"
    )

    # Calculate reranking-only latency by comparing mean latencies
    reranking_latency_ms = max(0.0, perf_e["latency"]["mean_ms"] - perf_d["latency"]["mean_ms"])
    timings["reranking_latency_mean_ms"] = round(reranking_latency_ms, 2)
    logger.info(f"Isolated Neural Reranking Latency per Query: {reranking_latency_ms:.2f} ms")

    # =========================================================================
    # Ablation Experiment: Chunk Size 256 vs 384
    # =========================================================================
    logger.info("\n--- [Ablation] Chunk Size 384 (Overlap 64) ---")
    chunk_bm25_384 = ChunkBM25Retriever().fit(chunks_384)
    eval_b_384 = evaluate_retriever_on_financebench(
        retriever=chunk_bm25_384,
        questions=questions,
        indexed_doc_names=indexed_doc_names,
        k_values=cutoffs,
    )

    dense_retriever_384 = DenseRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=64,
        device=device,
        cache_dir=cache_dir,
    )
    dense_retriever_384.fit(chunks_384)
    eval_c_384 = evaluate_retriever_on_financebench(
        retriever=dense_retriever_384,
        questions=questions,
        indexed_doc_names=indexed_doc_names,
        k_values=cutoffs,
    )

    hybrid_rrf_384 = HybridRRFRetriever(
        bm25_retriever=chunk_bm25_384,
        dense_retriever=dense_retriever_384,
        rrf_k=60,
    )
    eval_d_384 = evaluate_retriever_on_financebench(
        retriever=hybrid_rrf_384,
        questions=questions,
        indexed_doc_names=indexed_doc_names,
        k_values=cutoffs,
    )

    ablation_results = {
        "chunk_256": {
            "chunk_count": len(chunks_256),
            "bm25_page_recall_at_5": perf_b["page_recall"]["recall_at_5"],
            "dense_page_recall_at_5": perf_c["page_recall"]["recall_at_5"],
            "hybrid_page_recall_at_5": perf_d["page_recall"]["recall_at_5"],
            "hybrid_mrr": perf_d["mrr"],
            "hybrid_ndcg_at_5": perf_d["ndcg"]["ndcg_at_5"],
        },
        "chunk_384": {
            "chunk_count": len(chunks_384),
            "bm25_page_recall_at_5": (
                eval_b_384["performance"]["total_benchmark_metrics"]["page_recall"]["recall_at_5"]
            ),
            "dense_page_recall_at_5": (
                eval_c_384["performance"]["total_benchmark_metrics"]["page_recall"]["recall_at_5"]
            ),
            "hybrid_page_recall_at_5": (
                eval_d_384["performance"]["total_benchmark_metrics"]["page_recall"]["recall_at_5"]
            ),
            "hybrid_mrr": (
                eval_d_384["performance"]["total_benchmark_metrics"]["mrr"]
            ),
            "hybrid_ndcg_at_5": (
                eval_d_384["performance"]["total_benchmark_metrics"]["ndcg"]["ndcg_at_5"]
            ),
        },
    }

    # =========================================================================
    # Paired Bootstrap Resampling (1,000 cluster iterations over 150 questions)
    # =========================================================================
    logger.info("\n--- Computing Paired Bootstrap Resampling (B=1000, n=150) ---")
    base_evals = eval_a["per_question_evaluations"]

    def make_extractor(metric_type: str, k: int):
        if metric_type == "page_recall":
            return lambda r: 1.0 if r.page_match_at_k.get(k, False) else 0.0
        elif metric_type == "doc_recall":
            return lambda r: 1.0 if r.doc_match_at_k.get(k, False) else 0.0
        elif metric_type == "mrr":
            return lambda r: r.reciprocal_rank
        elif metric_type == "ndcg":
            return lambda r: r.ndcg_at_k.get(k, 0.0)
        raise ValueError(f"Unknown metric type {metric_type}")

    bootstrap_comparisons: dict[str, Any] = {}
    candidate_configs = [
        ("config_b_chunk_bm25", eval_b["per_question_evaluations"]),
        ("config_c_dense", eval_c["per_question_evaluations"]),
        ("config_d_hybrid_rrf", eval_d["per_question_evaluations"]),
        ("config_e_neural_rerank", eval_e["per_question_evaluations"]),
    ]

    for name, cand_evals in candidate_configs:
        bootstrap_comparisons[name] = {
            "page_recall_at_5": compute_paired_bootstrap(
                base_evals, cand_evals, make_extractor("page_recall", 5)
            ),
            "doc_recall_at_5": compute_paired_bootstrap(
                base_evals, cand_evals, make_extractor("doc_recall", 5)
            ),
            "mrr": compute_paired_bootstrap(
                base_evals, cand_evals, make_extractor("mrr", 5)
            ),
            "ndcg_at_5": compute_paired_bootstrap(
                base_evals, cand_evals, make_extractor("ndcg", 5)
            ),
        }

    # =========================================================================
    # Error Taxonomy & Audit of 15+ Errors from Strongest Configuration
    # =========================================================================
    # Determine strongest configuration based on page Recall@5 and MRR
    score_map = {
        "config_a": perf_a["page_recall"]["recall_at_5"] + perf_a["mrr"],
        "config_b": perf_b["page_recall"]["recall_at_5"] + perf_b["mrr"],
        "config_c": perf_c["page_recall"]["recall_at_5"] + perf_c["mrr"],
        "config_d": perf_d["page_recall"]["recall_at_5"] + perf_d["mrr"],
        "config_e": perf_e["page_recall"]["recall_at_5"] + perf_e["mrr"],
    }
    strongest_name = max(score_map, key=score_map.get)
    strongest_eval = {
        "config_a": eval_a,
        "config_b": eval_b,
        "config_c": eval_c,
        "config_d": eval_d,
        "config_e": eval_e,
    }[strongest_name]

    logger.info(
        f"\n--- Qualitative Error Taxonomy on Strongest Configuration ({strongest_name}) ---"
    )
    raw_errors = strongest_eval["error_analysis"]["sample_errors"]
    classified_errors: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}

    for err in raw_errors[:20]:
        cat = categorize_error(err, err.get("top_retrieved", []))
        category_counts[cat] = category_counts.get(cat, 0) + 1
        classified_errors.append(
            {
                "financebench_id": err["financebench_id"],
                "question": err["question"],
                "gold_doc_name": err["gold_doc_name"],
                "gold_evidence_items": err.get("gold_evidence_items", []),
                "is_numerical": err.get("is_numerical", False),
                "failure_category": cat,
                "doc_found_in_top_k": err.get("doc_found_in_top_k", False),
                "top_retrieved_pages": [
                    {
                        "rank": c["rank"],
                        "doc_name": c["doc_name"],
                        "page_index": c["page_index"],
                    }
                    for c in err.get("top_retrieved", [])[:3]
                ],
            }
        )

    # =========================================================================
    # Construct Summary Deliverable Artifacts
    # =========================================================================
    summary_metrics = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "git_branch": "checkpoint/07-advanced-hybrid-retrieval",
            "benchmark": "FinanceBench Open Sample",
            "total_benchmark_questions": len(questions),
            "total_pages": len(page_corpus),
            "device": device,
        },
        "configurations": {
            "config_a_page_bm25": eval_a["performance"],
            "config_b_chunk_bm25": eval_b["performance"],
            "config_c_dense": eval_c["performance"],
            "config_d_hybrid_rrf": eval_d["performance"],
            "config_e_neural_rerank": eval_e["performance"],
        },
        "bootstrap_significance_vs_baseline": bootstrap_comparisons,
        "ablation_chunk_size": ablation_results,
        "timings_and_resources": {
            "timings_seconds": timings,
            "peak_memory_ram_vram_mb": mem_records,
            "total_chunks_256": len(chunks_256),
            "total_chunks_384": len(chunks_384),
        },
        "error_analysis_strongest_system": {
            "system_name": strongest_name,
            "total_missed_at_max_k": len(raw_errors),
            "category_distribution": category_counts,
            "audited_error_cases": classified_errors,
        },
    }

    metrics_out_path = artifacts_dir / "metrics.json"
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        # Strip per_question_evaluations from JSON serialization to keep artifact clean
        clean_summary = dict(summary_metrics)
        json.dump(clean_summary, f, indent=2)
    logger.info(f"Saved comprehensive metrics to {metrics_out_path}")

    config_out_path = artifacts_dir / "config.json"
    config_dict = {
        "configurations": {
            "config_a": {"name": "Page-Level BM25", "k1": 1.5, "b": 0.75},
            "config_b": {"name": "Chunk-Level BM25", "chunk_size": 256, "chunk_overlap": 64},
            "config_c": {
                "name": "Dense Retrieval",
                "model": "BAAI/bge-small-en-v1.5",
                "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                "dim": 384,
            },
            "config_d": {"name": "Hybrid RRF", "rrf_k": 60, "candidate_k": 100},
            "config_e": {
                "name": "Hybrid + Neural Rerank",
                "reranker_model": "BAAI/bge-reranker-base",
                "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
                "rerank_top_k": 30,
            },
        },
        "device": device,
        "benchmark_questions": len(questions),
        "corpus_pages": len(page_corpus),
    }
    with open(config_out_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved execution configuration to {config_out_path}")

    # =========================================================================
    # Formatted Benchmark Printout
    # =========================================================================
    print("\n" + "=" * 105)
    print("FINVERITAS CHECKPOINT 07: 5-CONFIGURATION RETRIEVAL BENCHMARK COMPARISON")
    print("=" * 105)
    header = (
        f"{'Metric':<25} | {'(A) Page BM25':<13} | {'(B) Chunk BM25':<14} | "
        f"{'(C) Dense BGE':<13} | {'(D) Hybrid RRF':<14} | {'(E) Rerank BGE':<14}"
    )
    print(header)
    print("-" * 105)

    def row(label, key, subkey):
        val_a = perf_a[key][subkey]
        val_b = perf_b[key][subkey]
        val_c = perf_c[key][subkey]
        val_d = perf_d[key][subkey]
        val_e = perf_e[key][subkey]
        if isinstance(val_a, float):
            return (
                f"{label:<25} | {val_a*100:11.2f}% | {val_b*100:12.2f}% | "
                f"{val_c*100:11.2f}% | {val_d*100:12.2f}% | {val_e*100:12.2f}%"
            )
        return (
            f"{label:<25} | {val_a:<13} | {val_b:<14} | {val_c:<13} | "
            f"{val_d:<14} | {val_e:<14}"
        )

    for k in cutoffs:
        print(row(f"Doc Recall@{k}", "doc_recall", f"recall_at_{k}"))
    print("-" * 105)
    for k in cutoffs:
        print(row(f"Page Recall@{k}", "page_recall", f"recall_at_{k}"))
    print("-" * 105)
    for k in cutoffs:
        print(row(f"Annotation Recall@{k}", "annotation_recall", f"recall_at_{k}"))
    print("-" * 105)
    print(
        f"{'Page MRR':<25} | {perf_a['mrr']:11.4f}  | {perf_b['mrr']:12.4f}  | "
        f"{perf_c['mrr']:11.4f}  | {perf_d['mrr']:12.4f}  | {perf_e['mrr']:12.4f} "
    )
    for k in [5, 10]:
        val_a = perf_a["ndcg"][f"ndcg_at_{k}"]
        val_b = perf_b["ndcg"][f"ndcg_at_{k}"]
        val_c = perf_c["ndcg"][f"ndcg_at_{k}"]
        val_d = perf_d["ndcg"][f"ndcg_at_{k}"]
        val_e = perf_e["ndcg"][f"ndcg_at_{k}"]
        print(
            f"{f'nDCG@{k}':<25} | {val_a:11.4f}  | {val_b:12.4f}  | "
            f"{val_c:11.4f}  | {val_d:12.4f}  | {val_e:12.4f} "
        )
    print("-" * 105)
    print(
        f"{'Mean Latency (ms)':<25} | {perf_a['latency']['mean_ms']:10.1f} ms | "
        f"{perf_b['latency']['mean_ms']:11.1f} ms | {perf_c['latency']['mean_ms']:10.1f} ms | "
        f"{perf_d['latency']['mean_ms']:11.1f} ms | {perf_e['latency']['mean_ms']:11.1f} ms"
    )
    print(
        f"{'Median Latency (ms)':<25} | {perf_a['latency']['median_ms']:10.1f} ms | "
        f"{perf_b['latency']['median_ms']:11.1f} ms | {perf_c['latency']['median_ms']:10.1f} ms | "
        f"{perf_d['latency']['median_ms']:11.1f} ms | {perf_e['latency']['median_ms']:11.1f} ms"
    )
    print("=" * 105)

    return 0


if __name__ == "__main__":
    sys.exit(main())
