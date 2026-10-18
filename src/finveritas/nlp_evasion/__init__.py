"""
FinVeritas NLP Evasion Module:
Handles earnings call dataset ingestion, question-evasion classification,
hedging calibration, and claim extraction.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finveritas.nlp_evasion.baselines import (
        MajorityClassBaseline,
        ModelMetrics,
        build_tfidf_lr_pipeline,
        create_leakage_aware_split,
        evaluate_model,
        normalize_question_for_grouping,
        run_baseline_experiments,
    )
    from finveritas.nlp_evasion.dataset_audit import (
        DatasetAuditResult,
        audit_parquet_dataset,
        download_evasionbench_parquet,
    )

__all__ = [
    # Dataset Audit
    "DatasetAuditResult",
    "audit_parquet_dataset",
    "download_evasionbench_parquet",
    # Baselines
    "MajorityClassBaseline",
    "ModelMetrics",
    "build_tfidf_lr_pipeline",
    "create_leakage_aware_split",
    "evaluate_model",
    "normalize_question_for_grouping",
    "run_baseline_experiments",
]


def __getattr__(name: str):
    if name in (
        "DatasetAuditResult",
        "audit_parquet_dataset",
        "download_evasionbench_parquet",
    ):
        from finveritas.nlp_evasion import dataset_audit

        return getattr(dataset_audit, name)
    if name in (
        "MajorityClassBaseline",
        "ModelMetrics",
        "build_tfidf_lr_pipeline",
        "create_leakage_aware_split",
        "evaluate_model",
        "normalize_question_for_grouping",
        "run_baseline_experiments",
    ):
        from finveritas.nlp_evasion import baselines

        return getattr(baselines, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
