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
import math
import time
from typing import Any

import numpy as np

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
    ndcg_at_k: dict[int, float] = dataclasses.field(default_factory=dict)
    annotation_match_at_k: dict[int, int] = dataclasses.field(default_factory=dict)
    total_annotations: int = 0
    is_numerical: bool = False
    gold_evidence_items: list[tuple[str, int]] = dataclasses.field(default_factory=list)


NUMERICAL_KEYWORDS = {
    "ratio",
    "margin",
    "percentage",
    "dollar",
    "ebitda",
    "revenue",
    "expenditure",
    "capex",
    "debt",
    "operating income",
    "net income",
    "eps",
    "shares",
    "growth",
    "average",
    "total",
    "inventory",
    "cash",
    "depreciation",
    "amortization",
    "cagr",
    "dividend",
    "interest",
    "liabilities",
    "assets",
    "equity",
    "p/e",
    "gross profit",
    "operating expenses",
    "effective tax",
    "return on",
    "cost of",
    "working capital",
    "free cash flow",
    "rnd",
    "r&d",
    "basis points",
    "bps",
    "percent",
    "$",
    "%",
}


def is_numerical_question(question: str) -> bool:
    """Classifies whether a financial query is numerical/table-intensive or conceptual prose."""
    ql = question.lower()
    return any(kw in ql for kw in NUMERICAL_KEYWORDS)


def compute_ndcg_at_k(
    retrieved_pairs: list[tuple[str, int]],
    gold_pairs: set[tuple[str, int]],
    k: int,
) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@K) under binary relevance."""
    if not gold_pairs or k <= 0:
        return 0.0

    dcg = 0.0
    for i, pair in enumerate(retrieved_pairs[:k], start=1):
        if pair in gold_pairs:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(gold_pairs))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


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
        k_values = [1, 3, 5, 10]
    max_k = max(k_values)

    query_results: list[QueryEvaluationResult] = []
    total_benchmark_count = len(questions)

    for q in questions:
        q_id = str(q.get("financebench_id", "unknown"))
        q_text = str(q.get("question", "")).strip()
        gold_doc = str(q.get("doc_name", "")).strip()
        is_num = is_numerical_question(q_text)

        # Resolve all gold evidence annotations correctly preserving document and page
        gold_evidence_items: list[tuple[str, int]] = []
        gold_pages: list[int] = []
        for ev in q.get("evidence", []):
            ev_doc = str(
                ev.get("evidence_doc_name") or ev.get("doc_name") or gold_doc
            ).strip()
            page_num = ev.get("evidence_page_num")
            if page_num is not None:
                gold_evidence_items.append((ev_doc, int(page_num)))
                if ev_doc == gold_doc:
                    gold_pages.append(int(page_num))

        if not gold_pages and gold_evidence_items:
            gold_pages = [item[1] for item in gold_evidence_items]
        gold_pages = list(dict.fromkeys(gold_pages))
        gold_evidence_set = set(gold_evidence_items)

        # Check coverage
        is_covered = gold_doc in indexed_doc_names or any(
            doc in indexed_doc_names for doc, _ in gold_evidence_items
        )
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
                    ndcg_at_k={k: 0.0 for k in k_values},
                    annotation_match_at_k={k: 0 for k in k_values},
                    total_annotations=len(gold_evidence_items),
                    is_numerical=is_num,
                    gold_evidence_items=gold_evidence_items,
                )
            )
            continue

        # Execute retrieval with latency profiling
        start_t = time.perf_counter()
        retrieved: list[RetrievalResult] = retriever.retrieve(q_text, top_k=max_k)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        retrieved_dicts = [r.to_dict() for r in retrieved]
        retrieved_pairs = [(r.doc_name, r.page_index) for r in retrieved]

        # Calculate document and page matches at each k
        doc_matches: dict[int, bool] = {}
        page_matches: dict[int, bool] = {}
        ndcg_matches: dict[int, float] = {}
        annotation_matches: dict[int, int] = {}

        for k in k_values:
            top_k_items = retrieved[:k]
            top_k_pairs = retrieved_pairs[:k]

            # Doc match: any item matches gold_doc or any evidence doc
            doc_hit = any(
                (
                    item.doc_name == gold_doc
                    or any(item.doc_name == ev_doc for ev_doc, _ in gold_evidence_items)
                )
                for item in top_k_items
            )
            doc_matches[k] = doc_hit

            # Page match: any item matches an exact gold evidence page
            page_hit = any(pair in gold_evidence_set for pair in top_k_pairs)
            page_matches[k] = page_hit

            # nDCG@k
            ndcg_matches[k] = compute_ndcg_at_k(retrieved_pairs, gold_evidence_set, k)

            # Annotation match count: how many distinct gold annotations were retrieved in top k
            matched_ann_count = sum(1 for ann in gold_evidence_items if ann in set(top_k_pairs))
            annotation_matches[k] = matched_ann_count

        # Reciprocal Rank calculation (for first matching gold evidence page)
        rr = 0.0
        for rank_idx, pair in enumerate(retrieved_pairs, start=1):
            if pair in gold_evidence_set:
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
                ndcg_at_k=ndcg_matches,
                annotation_match_at_k=annotation_matches,
                total_annotations=len(gold_evidence_items),
                is_numerical=is_num,
                gold_evidence_items=gold_evidence_items,
            )
        )

    # Helper for aggregating subset metrics
    def compute_aggregates(subset: list[QueryEvaluationResult], denominator: int) -> dict[str, Any]:
        if denominator == 0:
            return {}

        total_ann = sum(r.total_annotations for r in subset)
        latencies = [r.latency_ms for r in subset if r.latency_ms > 0]
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        med_lat = float(np.median(latencies)) if latencies else 0.0

        agg: dict[str, Any] = {
            "denominator": denominator,
            "total_evidence_annotations": total_ann,
            "doc_recall": {},
            "page_recall": {},
            "annotation_recall": {},
            "ndcg": {},
            "mrr": round(sum(r.reciprocal_rank for r in subset) / denominator, 4),
            "latency": {
                "mean_ms": round(mean_lat, 2),
                "median_ms": round(med_lat, 2),
            },
        }

        for k in k_values:
            doc_hits = sum(1 for r in subset if r.doc_match_at_k.get(k, False))
            page_hits = sum(1 for r in subset if r.page_match_at_k.get(k, False))
            ann_hits = sum(r.annotation_match_at_k.get(k, 0) for r in subset)
            ndcg_sum = sum(r.ndcg_at_k.get(k, 0.0) for r in subset)

            agg["doc_recall"][f"recall_at_{k}"] = round(doc_hits / denominator, 4)
            agg["page_recall"][f"recall_at_{k}"] = round(page_hits / denominator, 4)
            agg["annotation_recall"][f"recall_at_{k}"] = (
                round(ann_hits / total_ann, 4) if total_ann > 0 else 0.0
            )
            agg["ndcg"][f"ndcg_at_{k}"] = round(ndcg_sum / denominator, 4)

        return agg

    covered_results = [r for r in query_results if r.is_covered]
    n_covered = len(covered_results)
    n_uncovered = total_benchmark_count - n_covered

    total_metrics = compute_aggregates(query_results, total_benchmark_count)

    # Subsets: numerical vs prose
    num_results = [r for r in query_results if r.is_numerical]
    prose_results = [r for r in query_results if not r.is_numerical]

    numerical_metrics = compute_aggregates(num_results, len(num_results)) if num_results else {}
    prose_metrics = compute_aggregates(prose_results, len(prose_results)) if prose_results else {}

    # Collect error cases for qualitative audit
    error_cases = []
    for r in query_results:
        if not r.page_match_at_k.get(max_k, False):
            error_cases.append(
                {
                    "financebench_id": r.financebench_id,
                    "question": r.question,
                    "gold_doc_name": r.gold_doc_name,
                    "gold_evidence_items": r.gold_evidence_items,
                    "is_numerical": r.is_numerical,
                    "doc_found_in_top_k": r.doc_match_at_k.get(max_k, False),
                    "top_retrieved": [
                        {
                            "rank": item["rank"],
                            "doc_name": item["doc_name"],
                            "page_index": item["page_index"],
                            "score": item["score"],
                            "text_snippet": item["text_snippet"][:150],
                        }
                        for item in r.retrieved_results[:5]
                    ],
                }
            )

    covered_metrics = (
        compute_aggregates(covered_results, n_covered) if covered_results else {}
    )

    return {
        "benchmark_summary": {
            "total_benchmark_questions": total_benchmark_count,
            "covered_questions": n_covered,
            "uncovered_questions": n_uncovered,
            "coverage_fraction": round(n_covered / total_benchmark_count, 4)
            if total_benchmark_count > 0
            else 0.0,
            "numerical_questions_count": len(num_results),
            "prose_questions_count": len(prose_results),
        },
        "performance": {
            "total_benchmark_metrics": total_metrics,
            "covered_subset_metrics": covered_metrics,
            "eligible_subset_metrics": covered_metrics,
            "numerical_subset_metrics": numerical_metrics,
            "prose_subset_metrics": prose_metrics,
        },
        "error_analysis": {
            "missed_at_max_k_count": len(error_cases),
            "sample_errors": error_cases,
        },
        "per_question_evaluations": query_results,
    }


def compute_paired_bootstrap(
    baseline_evals: list[QueryEvaluationResult],
    candidate_evals: list[QueryEvaluationResult],
    metric_extractor: Any,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Computes paired bootstrap resampling over question clusters (n=150 questions).
    Returns mean candidate metric, mean baseline metric, mean difference (candidate - baseline),
    95% confidence interval [2.5%, 97.5%], and empirical p-value P(diff <= 0).
    """
    if len(baseline_evals) != len(candidate_evals):
        raise ValueError("Baseline and candidate must evaluate the same number of questions.")

    rng = np.random.default_rng(seed)
    n = len(baseline_evals)

    base_vals = np.array([metric_extractor(r) for r in baseline_evals], dtype=np.float64)
    cand_vals = np.array([metric_extractor(r) for r in candidate_evals], dtype=np.float64)

    observed_base = float(np.mean(base_vals))
    observed_cand = float(np.mean(cand_vals))
    observed_diff = observed_cand - observed_base

    boot_diffs = []
    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        b_base = np.mean(base_vals[boot_idx])
        b_cand = np.mean(cand_vals[boot_idx])
        boot_diffs.append(b_cand - b_base)

    boot_diffs_arr = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs_arr, 2.5))
    ci_upper = float(np.percentile(boot_diffs_arr, 97.5))
    p_value = float(np.mean(boot_diffs_arr <= 0.0))

    return {
        "candidate_mean": round(observed_cand, 4),
        "baseline_mean": round(observed_base, 4),
        "mean_difference": round(observed_diff, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "p_value": round(p_value, 4),
        "statistically_significant": ci_lower > 0.0,
    }

