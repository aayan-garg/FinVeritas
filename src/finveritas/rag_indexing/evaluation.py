"""
FinVeritas: Evaluation Framework for FinanceBench Evidence Retrieval.

Evaluates lexical retrieval engines against official FinanceBench annotations:
- Gold document Recall@1, Recall@3, Recall@5
- Gold evidence page Recall@1, Recall@3, Recall@5 (0-indexed page matching)
- Mean Reciprocal Rank (MRR) for gold evidence pages
- Retrieval latency profiling
- Dual denominator metrics: eligible subset vs. total benchmark (n=150)
- Failure mode categorization for qualitative error audits
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from finveritas.rag_indexing.retrievers import RetrievalResult


@dataclasses.dataclass(frozen=True)
class QueryEvaluationResult:
    """Individual evaluation record for a benchmark question."""

    financebench_id: str
    question: str
    gold_doc_name: str
    gold_pages: list[int]  # 0-indexed page numbers
    is_covered: bool
    uncovered_reason: str
    latency_ms: float
    retrieved_results: list[dict[str, Any]]
    doc_match_at_k: dict[int, bool]
    page_match_at_k: dict[int, bool]
    reciprocal_rank: float


def evaluate_retriever_on_financebench(
    retriever: Any,
    questions: list[dict[str, Any]],
    indexed_doc_names: set[str],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluates a fitted retriever across FinanceBench questions.
    Returns aggregated metrics, per-question breakdown, and error analysis cases.
    """
    if k_values is None:
        k_values = [1, 3, 5]
    max_k = max(k_values)

    query_results: list[QueryEvaluationResult] = []
    total_benchmark_count = len(questions)

    for q in questions:
        q_id = str(q.get("financebench_id", "unknown"))
        q_text = str(q.get("question", "")).strip()
        gold_doc = str(q.get("doc_name", "")).strip()

        # Extract gold page references from evidence
        gold_pages = []
        for ev in q.get("evidence", []):
            page_num = ev.get("evidence_page_num")
            if page_num is not None:
                gold_pages.append(int(page_num))
        # Deduplicate while preserving order
        gold_pages = list(dict.fromkeys(gold_pages))

        # Check coverage
        is_covered = gold_doc in indexed_doc_names
        uncovered_reason = "" if is_covered else f"Source document '{gold_doc}' not indexed"

        if not is_covered:
            query_results.append(
                QueryEvaluationResult(
                    financebench_id=q_id,
                    question=q_text,
                    gold_doc_name=gold_doc,
                    gold_pages=gold_pages,
                    is_covered=False,
                    uncovered_reason=uncovered_reason,
                    latency_ms=0.0,
                    retrieved_results=[],
                    doc_match_at_k={k: False for k in k_values},
                    page_match_at_k={k: False for k in k_values},
                    reciprocal_rank=0.0,
                )
            )
            continue

        # Execute retrieval with latency profiling
        start_t = time.perf_counter()
        retrieved: list[RetrievalResult] = retriever.retrieve(q_text, top_k=max_k)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        retrieved_dicts = [r.to_dict() for r in retrieved]

        # Calculate document and page matches at each k
        doc_matches: dict[int, bool] = {}
        page_matches: dict[int, bool] = {}

        for k in k_values:
            top_k_items = retrieved[:k]
            # Doc match: any item matches gold_doc
            doc_hit = any(item.doc_name == gold_doc for item in top_k_items)
            doc_matches[k] = doc_hit

            # Page match: any item matches gold_doc and is in gold_pages
            page_hit = any(
                (item.doc_name == gold_doc and item.page_index in gold_pages)
                for item in top_k_items
            )
            page_matches[k] = page_hit

        # Reciprocal Rank calculation (for gold evidence page)
        rr = 0.0
        for rank_idx, item in enumerate(retrieved, start=1):
            if item.doc_name == gold_doc and item.page_index in gold_pages:
                rr = 1.0 / rank_idx
                break

        query_results.append(
            QueryEvaluationResult(
                financebench_id=q_id,
                question=q_text,
                gold_doc_name=gold_doc,
                gold_pages=gold_pages,
                is_covered=True,
                uncovered_reason="",
                latency_ms=elapsed_ms,
                retrieved_results=retrieved_dicts,
                doc_match_at_k=doc_matches,
                page_match_at_k=page_matches,
                reciprocal_rank=rr,
            )
        )

    # Aggregate metrics
    covered_results = [r for r in query_results if r.is_covered]
    n_covered = len(covered_results)
    n_uncovered = total_benchmark_count - n_covered

    def compute_aggregates(subset: list[QueryEvaluationResult], denominator: int) -> dict[str, Any]:
        if denominator == 0:
            return {}
        agg: dict[str, Any] = {
            "denominator": denominator,
            "doc_recall": {},
            "page_recall": {},
            "mrr": round(sum(r.reciprocal_rank for r in subset) / denominator, 4),
        }
        for k in k_values:
            doc_hits = sum(1 for r in subset if r.doc_match_at_k.get(k, False))
            page_hits = sum(1 for r in subset if r.page_match_at_k.get(k, False))
            agg["doc_recall"][f"recall_at_{k}"] = round(doc_hits / denominator, 4)
            agg["page_recall"][f"recall_at_{k}"] = round(page_hits / denominator, 4)
        return agg

    eligible_metrics = compute_aggregates(covered_results, n_covered) if n_covered > 0 else {}
    total_metrics = compute_aggregates(covered_results, total_benchmark_count)

    mean_latency_ms = (
        round(sum(r.latency_ms for r in covered_results) / n_covered, 2)
        if n_covered > 0
        else 0.0
    )

    # Collect error cases for qualitative audit (questions where gold page was missed at max_k)
    error_cases = []
    for r in covered_results:
        if not r.page_match_at_k.get(max_k, False):
            error_cases.append(
                {
                    "financebench_id": r.financebench_id,
                    "question": r.question,
                    "gold_doc_name": r.gold_doc_name,
                    "gold_pages": r.gold_pages,
                    "doc_found_in_top_k": r.doc_match_at_k.get(max_k, False),
                    "top_retrieved": [
                        {
                            "rank": item["rank"],
                            "doc_name": item["doc_name"],
                            "page_index": item["page_index"],
                            "score": item["score"],
                            "text_snippet": item["text_snippet"][:150],
                        }
                        for item in r.retrieved_results[:3]
                    ],
                }
            )

    return {
        "benchmark_summary": {
            "total_benchmark_questions": total_benchmark_count,
            "covered_questions": n_covered,
            "uncovered_questions": n_uncovered,
            "coverage_fraction": round(n_covered / total_benchmark_count, 4)
            if total_benchmark_count > 0
            else 0.0,
        },
        "performance": {
            "mean_retrieval_latency_ms": mean_latency_ms,
            "eligible_subset_metrics": eligible_metrics,
            "total_benchmark_metrics": total_metrics,
        },
        "error_analysis": {
            "missed_at_top_k_count": len(error_cases),
            "sample_errors": error_cases[:10],
        },
        "per_question_evaluations": [
            {
                "financebench_id": r.financebench_id,
                "is_covered": r.is_covered,
                "gold_doc_name": r.gold_doc_name,
                "gold_pages": r.gold_pages,
                "page_match_at_k": r.page_match_at_k,
                "doc_match_at_k": r.doc_match_at_k,
                "reciprocal_rank": round(r.reciprocal_rank, 4),
                "latency_ms": round(r.latency_ms, 2),
            }
            for r in query_results
        ],
    }
