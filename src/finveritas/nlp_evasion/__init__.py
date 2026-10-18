"""
FinVeritas NLP Evasion Module:
Handles earnings call dataset ingestion, question-evasion classification,
hedging calibration, and claim extraction.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finveritas.nlp_evasion.dataset_audit import (
        DatasetAuditResult,
        audit_parquet_dataset,
        download_evasionbench_parquet,
    )

__all__ = [
    "DatasetAuditResult",
    "audit_parquet_dataset",
    "download_evasionbench_parquet",
]


def __getattr__(name: str):
    if name in __all__:
        from finveritas.nlp_evasion import dataset_audit

        return getattr(dataset_audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
