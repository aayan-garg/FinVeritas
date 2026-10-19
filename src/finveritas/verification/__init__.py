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
from finveritas.verification.numeric_verifier import (
    KNOWN_ENTITY_MAPPINGS,
    MAX_PERMITTED_TOLERANCE,
    EvidenceCitation,
    EvidencePassage,
    FinancialClaim,
    ParsedEvidenceCandidate,
    VerificationResult,
    VerificationVerdict,
    canonicalize_metric,
    normalize_currency,
    normalize_period,
    normalize_scale,
    validate_entity_match,
    verify_claim,
)

__all__ = [
    "AnswerQualityRecord",
    "CANONICAL_CLASSES",
    "EvidenceCitation",
    "EvidencePassage",
    "FinancialClaim",
    "KNOWN_ENTITY_MAPPINGS",
    "LABEL_MAPPING",
    "MAX_PERMITTED_TOLERANCE",
    "ParsedEvidenceCandidate",
    "VerificationResult",
    "VerificationVerdict",
    "build_classifier_pipeline",
    "build_text_vectorizer",
    "canonicalize_metric",
    "compute_multiclass_metrics",
    "evaluate_grouped_baselines",
    "load_and_validate_results",
    "normalize_currency",
    "normalize_period",
    "normalize_scale",
    "paired_group_bootstrap",
    "validate_entity_match",
    "verify_claim",
]
