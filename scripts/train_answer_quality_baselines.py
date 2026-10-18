#!/usr/bin/env python3
"""
FinVeritas: Train and Evaluate Financial Answer-Quality Baselines.

Runs 5-fold group-aware cross-validation on the 2,400 human-annotated FinanceBench
model completions across four baseline configurations:
  A. Majority-Class Baseline
  B. Question-Only TF-IDF + Logistic Regression
  C. Answer-Only TF-IDF + Logistic Regression
  D. Question + Answer TF-IDF + Logistic Regression

Saves:
  - artifacts/answer_quality/metrics.json
  - artifacts/answer_quality/config.json
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

from finveritas.verification import (  # noqa: E402
    evaluate_grouped_baselines,
    load_and_validate_results,
)

logger = logging.getLogger("train_answer_quality")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinVeritas Answer-Quality Baseline Evaluation (5-Fold Grouped CV)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/raw/financebench/results"),
        help="Directory containing FinanceBench results/*.jsonl files.",
    )
    parser.add_argument(
        "--open-source-jsonl",
        type=Path,
        default=Path("data/raw/financebench/financebench_open_source.jsonl"),
        help="Path to official financebench_open_source.jsonl metadata.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/answer_quality"),
        help="Directory to save output metrics.json and config.json.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of grouped cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational log messages.",
    )

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate data
    logger.info(f"Loading and validating records from {args.results_dir}...")
    records = load_and_validate_results(
        results_dir=args.results_dir,
        open_source_path=args.open_source_jsonl,
    )
    logger.info(f"Successfully loaded and validated {len(records)} records.")

    # 2. Execute 5-fold grouped CV
    logger.info(
        f"Running {args.n_splits}-fold grouped cross-validation (seed={args.random_state})..."
    )
    results = evaluate_grouped_baselines(
        records=records,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    # 3. Save config.json
    config_payload = {
        "task": "financial_answer_quality_classification",
        "description": (
            "Multi-class classification of financial model answers into "
            "'correct', 'incorrect', and 'refusal' using 5-fold grouped cross-validation."
        ),
        "dataset": {
            "source": "PatronusAI/financebench (results/ directory)",
            "total_records": len(records),
            "unique_question_groups": results["dataset_summary"]["unique_question_groups"],
            "classes": ["correct", "incorrect", "refusal"],
            "label_distribution": results["dataset_summary"]["label_distribution"],
        },
        "validation_strategy": {
            "type": "StratifiedGroupKFold",
            "n_splits": args.n_splits,
            "grouping_key": "financebench_id",
            "random_state": args.random_state,
            "leakage_prevention": (
                "Zero question-ID overlap asserted across all folds. "
                "No gold reference answers, evidence text, or page annotations exposed to models."
            ),
        },
        "model_configurations": {
            "majority": {
                "type": "MajorityClassPredictor",
                "predicted_class": "correct",
            },
            "question_only": {
                "type": "LogisticRegression",
                "input_features": "question_text",
                "vectorizer": "FeatureUnion(Word_TFIDF(1,2) + Char_TFIDF(3,5))",
                "class_weight": "balanced",
                "max_iter": 1000,
                "C": 1.0,
            },
            "answer_only": {
                "type": "LogisticRegression",
                "input_features": "model_answer_text",
                "vectorizer": "FeatureUnion(Word_TFIDF(1,2) + Char_TFIDF(3,5))",
                "class_weight": "balanced",
                "max_iter": 1000,
                "C": 1.0,
            },
            "question_plus_answer": {
                "type": "LogisticRegression",
                "input_features": "question_text + model_answer_text",
                "vectorizer": "FeatureUnion(Word_TFIDF(1,2) + Char_TFIDF(3,5))",
                "class_weight": "balanced",
                "max_iter": 1000,
                "C": 1.0,
            },
        },
    }

    config_path = artifacts_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    logger.info(f"Saved configuration artifact to {config_path}")

    # 4. Save metrics.json
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics artifact to {metrics_path}")

    # 5. Print summary table
    print("\n" + "=" * 80)
    print("FINVERITAS ANSWER-QUALITY CLASSIFICATION BENCHMARK (5-FOLD GROUPED CV)")
    print("=" * 80)
    print(f"Total Evaluated Records:       {len(records)}")
    print(f"Unique Question Groups:        {results['dataset_summary']['unique_question_groups']}")
    print(f"Label Counts:                  {results['dataset_summary']['label_distribution']}")
    print("-" * 80)
    header = (
        f"{'Configuration':<25} | {'Accuracy':<10} | {'Macro-F1':<10} | "
        f"{'Weighted-F1':<12} | {'Runtime (s)':<10}"
    )
    print(header)
    print("-" * 80)

    for m_name, m_data in results["models"].items():
        m_met = m_data["overall_metrics"]
        row = (
            f"{m_name:<25} | "
            f"{m_met['accuracy']:<10.4f} | "
            f"{m_met['macro_f1']:<10.4f} | "
            f"{m_met['weighted_f1']:<12.4f} | "
            f"{m_data['runtime_seconds']:<10.2f}"
        )
        print(row)
    print("=" * 80)

    print("\nPer-Class Breakdown (Answer-Only Baseline):")
    ans_metrics = results["models"]["answer_only"]["overall_metrics"]["per_class"]
    per_class_hdr = (
        f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}"
    )
    print(per_class_hdr)
    print("-" * 60)
    for c_name, c_data in ans_metrics.items():
        print(
            f"{c_name:<12} | "
            f"{c_data['precision']:<10.4f} | "
            f"{c_data['recall']:<10.4f} | "
            f"{c_data['f1']:<10.4f} | "
            f"{c_data['support']:<10d}"
        )
    print("=" * 60)

    print("\nPaired Group Bootstrap Significance (Answer-Only vs Alternatives):")
    for comp_name, boot in results["paired_group_bootstraps"].items():
        print(
            f"  {comp_name:<35}: "
            f"Delta Acc: {boot['mean_delta_accuracy']:+.4f} (95% CI: {boot['ci_95_accuracy']}), "
            f"Delta F1: {boot['mean_delta_macro_f1']:+.4f} (95% CI: {boot['ci_95_macro_f1']}), "
            f"p(Acc)={boot['p_value_accuracy']:.4f}"
        )
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
