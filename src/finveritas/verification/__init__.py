"""
FinVeritas Verification Module:
Numerical calculation validation, sandboxed program-aided verification,
and Natural Language Inference (NLI) claim verification.
"""

from finveritas.verification.answer_quality import (
    CANONICAL_CLASSES,
    LABEL_MAPPING,
    AnswerQualityRecord,
    build_classifier_pipeline,
    build_text_vectorizer,
    compute_multiclass_metrics,
    evaluate_grouped_baselines,
    load_and_validate_results,
    paired_group_bootstrap,
)

__all__ = [
    "AnswerQualityRecord",
    "CANONICAL_CLASSES",
    "LABEL_MAPPING",
    "build_classifier_pipeline",
    "build_text_vectorizer",
    "compute_multiclass_metrics",
    "evaluate_grouped_baselines",
    "load_and_validate_results",
    "paired_group_bootstrap",
]
