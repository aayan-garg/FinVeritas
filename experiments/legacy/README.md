# Legacy Experiments & Exploratory Artifacts

This directory contains exploratory, pre-bootstrap artifacts preserved for provenance, historical context, and reference during subsequent implementation phases.

## ⚠️ Important Notice

- **Not Supported API:** The scripts and tests in this directory are exploratory research prototypes and are **not** part of the supported FinVeritas training, inference, or testing pipeline.
- **Excluded from Default Test Suite:** These files require heavyweight machine learning libraries (such as PyTorch, Transformers, and Scikit-Learn) and are intentionally isolated from the root bootstrap test suite (`pytest tests/`).
- **Do Not Execute in Checkpoint 00/00A:** These artifacts must not be run during bootstrap compliance audits or CI sanity checks.

## Inventory of Legacy Artifacts

### 1. `train_baseline.py`
- **Origin:** Root-level exploratory spike created during initial EvasionBench investigation.
- **Description:** Prototype sequence classification cross-encoder script exploring DeBERTa-v3-small hyperparameter configurations, class-weighted cross-entropy loss, and 6GB VRAM memory bounds.
- **Status:** Relocated from `scripts/train_baseline.py` to `experiments/legacy/train_baseline.py` under ADR-0002 / Checkpoint 00A compliance. Reusable components will be formally reviewed and implemented into `src/finveritas/nlp_evasion/` in Checkpoint 01.

### 2. `test_baseline_acceptance.py`
- **Origin:** Root-level unit test file (`acceptance_tests.py`), previously moved to `tests/unit/test_baseline_acceptance.py`.
- **Description:** Acceptance tests for label mappings, class-weight inverse frequency math, and training hyperparameter constraints. Strongly coupled to PyTorch and the legacy training prototype.
- **Status:** Relocated from `tests/unit/test_baseline_acceptance.py` to `experiments/legacy/test_baseline_acceptance.py` to ensure root test suites remain lightweight and free of heavyweight ML/GPU dependencies during repository bootstrap.
