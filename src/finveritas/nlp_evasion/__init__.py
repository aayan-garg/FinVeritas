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
        load_baseline_model,
        normalize_question_for_grouping,
        predict_evasion,
        run_baseline_experiments,
        train_and_save_baseline_model,
    )
    from finveritas.nlp_evasion.dataset_audit import (
        DatasetAuditResult,
        audit_parquet_dataset,
        download_evasionbench_parquet,
    )
    from finveritas.nlp_evasion.transformer_classifier import (
        DEFAULT_MAX_LENGTH,
        DEFAULT_MODEL_NAME,
        DISCLAIMER_TEXT,
        EvasionQAPairDataset,
        TransformerModelMetrics,
        collate_qa_batch,
        compute_metrics,
        evaluate_transformer_model,
        load_and_split_evasion_data,
        predict_evasion_transformer,
        run_forward_backward_smoke_test,
        train_transformer_model,
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
    "load_baseline_model",
    "normalize_question_for_grouping",
    "predict_evasion",
    "run_baseline_experiments",
    "train_and_save_baseline_model",
    # Transformer Evasion
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_NAME",
    "DISCLAIMER_TEXT",
    "EvasionQAPairDataset",
    "TransformerModelMetrics",
    "collate_qa_batch",
    "compute_metrics",
    "evaluate_transformer_model",
    "load_and_split_evasion_data",
    "predict_evasion_transformer",
    "run_forward_backward_smoke_test",
    "train_transformer_model",
]

_TRANSFORMER_EXPORTS = {
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_NAME",
    "DISCLAIMER_TEXT",
    "EvasionQAPairDataset",
    "TransformerModelMetrics",
    "collate_qa_batch",
    "compute_metrics",
    "evaluate_transformer_model",
    "load_and_split_evasion_data",
    "predict_evasion_transformer",
    "run_forward_backward_smoke_test",
    "train_transformer_model",
}


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
        "load_baseline_model",
        "normalize_question_for_grouping",
        "predict_evasion",
        "run_baseline_experiments",
        "train_and_save_baseline_model",
    ):
        from finveritas.nlp_evasion import baselines

        return getattr(baselines, name)
    if name in _TRANSFORMER_EXPORTS:
        from finveritas.nlp_evasion import transformer_classifier

        return getattr(transformer_classifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
