#!/usr/bin/env python3
"""
FinVeritas: CLI for FinanceBench Evidence Retrieval Evaluation.

Evaluates and compares TF-IDF and BM25 page-level retrieval baselines on
FinanceBench's open-source 150-question benchmark.

Usage:
  # Download metadata and PDFs (if not already local), build corpus, and evaluate:
  python scripts/evaluate_financebench_retrieval.py --download

  # Run evaluation using already downloaded local corpus:
  python scripts/evaluate_financebench_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from finveritas.rag_indexing import (  # noqa: E402
    BM25Retriever,
    TfidfRetriever,
    build_page_corpus,
    download_financebench_metadata,
    download_financebench_pdfs,
    evaluate_retriever_on_financebench,
    load_financebench_metadata,
)

logger = logging.getLogger("evaluate_financebench_retrieval")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinVeritas FinanceBench Evidence Retrieval Baseline Evaluation (CPU-Only)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/financebench"),
        help="Directory containing FinanceBench metadata and PDFs.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/financebench_retrieval"),
        help="Directory to save derived metrics.json and config.json.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing metadata and referenced source PDFs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-K cutoff ranks to evaluate (default: 1 3 5).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose debug/info logging.",
    )

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    data_dir = args.data_dir
    artifacts_dir = args.artifacts_dir
    pdf_dir = data_dir / "pdfs"

    open_source_jsonl = data_dir / "financebench_open_source.jsonl"
    doc_info_jsonl = data_dir / "financebench_document_information.jsonl"

    # Step 1: Ensure metadata exists
    if not (open_source_jsonl.is_file() and doc_info_jsonl.is_file()):
        if args.download:
            logger.info("Downloading FinanceBench metadata...")
            download_financebench_metadata(data_dir)
        else:
            logger.error(
                f"Missing metadata in {data_dir}. Run with --download to acquire dataset files."
            )
            return 1

    questions, doc_info_map = load_financebench_metadata(
        open_source_path=open_source_jsonl,
        doc_info_path=doc_info_jsonl,
    )
    logger.info(
        f"Loaded {len(questions)} benchmark questions across "
        f"{len(doc_info_map)} cataloged documents."
    )

    needed_doc_names = set(q["doc_name"] for q in questions if "doc_name" in q)
    logger.info(
        f"Identified {len(needed_doc_names)} unique source PDFs required for the 150 questions."
    )

    # Step 2: Ensure PDFs exist
    if args.download:
        logger.info(f"Checking and downloading required PDFs to {pdf_dir}...")
        dl_stats = download_financebench_pdfs(
            needed_doc_names=needed_doc_names,
            pdf_dir=pdf_dir,
            timeout=30,
            max_retries=3,
        )
        logger.info(
            f"PDF Status: {dl_stats['existing_count']} existing, "
            f"{dl_stats['downloaded_count']} downloaded, "
            f"{dl_stats['failed_count']} failed."
        )

    # Step 3: Build page-level corpus
    logger.info("Extracting and building page-level corpus from local PDFs...")
    corpus, corpus_stats = build_page_corpus(
        pdf_dir=pdf_dir,
        doc_info_map=doc_info_map,
        needed_doc_names=needed_doc_names,
    )

    logger.info(
        f"Corpus built: {corpus_stats['pdfs_successfully_parsed']}/"
        f"{corpus_stats['total_pdfs_targeted']} PDFs parsed, "
        f"{corpus_stats['total_pages_extracted']} total pages "
        f"({corpus_stats['non_empty_pages_count']} content, "
        f"{corpus_stats['empty_pages_count']} empty)."
    )

    if not corpus:
        logger.error("Page corpus is empty. Cannot evaluate retrieval.")
        return 1

    indexed_docs = set(p.doc_name for p in corpus)

    # Step 4: Fit and Evaluate TF-IDF Retriever
    logger.info("Fitting TF-IDF retriever on page corpus...")
    tfidf = TfidfRetriever()
    tfidf.fit(corpus)
    logger.info("Evaluating TF-IDF retriever across questions...")
    tfidf_eval = evaluate_retriever_on_financebench(
        retriever=tfidf,
        questions=questions,
        indexed_doc_names=indexed_docs,
        k_values=args.top_k,
    )

    # Step 5: Fit and Evaluate BM25 Retriever
    logger.info("Fitting BM25 retriever on page corpus...")
    bm25 = BM25Retriever()
    bm25.fit(corpus)
    logger.info("Evaluating BM25 retriever across questions...")
    bm25_eval = evaluate_retriever_on_financebench(
        retriever=bm25,
        questions=questions,
        indexed_doc_names=indexed_docs,
        k_values=args.top_k,
    )

    # Step 6: Print Evaluation Results
    cov = tfidf_eval["benchmark_summary"]
    print("\n" + "=" * 76)
    print("FINVERITAS FINANCEBENCH EVIDENCE RETRIEVAL BENCHMARK RESULTS")
    print("=" * 76)
    print(f"Total Benchmark Questions:       {cov['total_benchmark_questions']}")
    print(f"Covered Questions (Indexed):     {cov['covered_questions']}")
    print(f"Uncovered Questions (Missing):   {cov['uncovered_questions']}")
    print(f"Coverage Fraction:               {cov['coverage_fraction']*100:.2f}%")
    print(f"Total Source PDFs Targeted:      {corpus_stats['total_pdfs_targeted']}")
    print(f"Source PDFs Successfully Parsed: {corpus_stats['pdfs_successfully_parsed']}")
    print(f"Total Page Corpus Size:          {corpus_stats['total_pages_extracted']} pages")
    print("-" * 76)

    # Table formatting
    print(f"{'Metric':<32} | {'TF-IDF (Cosine)':<18} | {'Okapi BM25':<18}")
    print("-" * 76)

    # Eligible subset metrics
    tf_elig = tfidf_eval["performance"]["eligible_subset_metrics"]
    bm_elig = bm25_eval["performance"]["eligible_subset_metrics"]
    tf_tot = tfidf_eval["performance"]["total_benchmark_metrics"]
    bm_tot = bm25_eval["performance"]["total_benchmark_metrics"]

    print("[Eligible Covered Subset Metrics]")
    for k in args.top_k:
        print(
            f"  Gold Doc Recall@{k:<15} | "
            f"{tf_elig['doc_recall'][f'recall_at_{k}']*100:6.2f}%            | "
            f"{bm_elig['doc_recall'][f'recall_at_{k}']*100:6.2f}%"
        )
    for k in args.top_k:
        print(
            f"  Gold Page Recall@{k:<14} | "
            f"{tf_elig['page_recall'][f'recall_at_{k}']*100:6.2f}%            | "
            f"{bm_elig['page_recall'][f'recall_at_{k}']*100:6.2f}%"
        )
    print(
        f"  Gold Page MRR                  | "
        f"{tf_elig['mrr']:6.4f}             | "
        f"{bm_elig['mrr']:6.4f}"
    )

    print("\n[Total Benchmark Denominator (n=150) Metrics]")
    for k in args.top_k:
        print(
            f"  Total Gold Page Recall@{k:<8} | "
            f"{tf_tot['page_recall'][f'recall_at_{k}']*100:6.2f}%            | "
            f"{bm_tot['page_recall'][f'recall_at_{k}']*100:6.2f}%"
        )
    print(
        f"  Total Gold Page MRR            | "
        f"{tf_tot['mrr']:6.4f}             | "
        f"{bm_tot['mrr']:6.4f}"
    )

    print("-" * 76)
    print(
        f"Mean Query Latency (ms):         | "
        f"{tfidf_eval['performance']['mean_retrieval_latency_ms']:6.2f} ms          | "
        f"{bm25_eval['performance']['mean_retrieval_latency_ms']:6.2f} ms"
    )
    print("=" * 76)

    # Step 7: Display Sample Qualitative Errors
    print("\n" + "=" * 76)
    print("QUALITATIVE ERROR AUDIT (FIRST 5 BM25 MISSED RETRIEVALS)")
    print("=" * 76)
    for idx, err in enumerate(bm25_eval["error_analysis"]["sample_errors"][:5], start=1):
        print(f"\n--- Error Case {idx}: [{err['financebench_id']}] ---")
        print(f"Question:      {err['question']}")
        print(f"Gold Document: {err['gold_doc_name']}")
        print(f"Gold Page(s):  {err['gold_pages']} (0-indexed)")
        print(f"Doc in Top-5:  {'YES' if err['doc_found_in_top_k'] else 'NO'}")
        print("Top Retrieved Candidates:")
        for cand in err["top_retrieved"]:
            print(
                f"  Rank {cand['rank']}: {cand['doc_name']} (Page {cand['page_index']}) - "
                f"Score: {cand['score']:.4f}\n"
                f"    Snippet: {cand['text_snippet'][:120]}..."
            )
    print("=" * 76 + "\n")

    # Step 8: Save JSON Artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "metrics.json"
    config_path = artifacts_dir / "config.json"

    metrics_payload = {
        "benchmark_summary": cov,
        "corpus_statistics": corpus_stats,
        "tfidf_baseline": tfidf_eval["performance"],
        "bm25_baseline": bm25_eval["performance"],
        "bm25_error_analysis": {
            "missed_at_top_5_count": bm25_eval["error_analysis"]["missed_at_top_k_count"],
            "inspected_error_cases": bm25_eval["error_analysis"]["sample_errors"][:5],
        },
    }

    config_payload = {
        "dataset": "FinanceBench (Open Source 150-Question Sample)",
        "sources": {
            "huggingface": "https://huggingface.co/datasets/PatronusAI/financebench",
            "github": "https://github.com/patronus-ai/financebench",
        },
        "license": "CC BY-NC 4.0",
        "top_k_ranks": args.top_k,
        "pdf_parser": "pypdf",
        "corpus_scope": {
            "target_documents": len(needed_doc_names),
            "parsed_documents": corpus_stats["pdfs_successfully_parsed"],
            "total_pages": corpus_stats["total_pages_extracted"],
        },
        "models": {
            "tfidf": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "similarity": "cosine",
            },
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "algorithm": "Okapi BM25",
            },
        },
        "disclaimer": (
            "Retrieval evaluation measures surface page-level evidence locating. "
            "It does not evaluate numerical reasoning or answer generation correctness."
        ),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved metrics artifact to {metrics_path}")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    logger.info(f"Saved config artifact to {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
