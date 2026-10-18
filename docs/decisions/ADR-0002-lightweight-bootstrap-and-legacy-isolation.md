# ADR-0002: Lightweight Bootstrap Environment & Legacy Spike Isolation

## Context & Problem Statement
Following initial bootstrap in Checkpoint 00, an audit revealed that:
1. `pyproject.toml` declared premature heavyweight machine learning dependencies (PyTorch, Transformers, pandas, PyArrow, scikit-learn, Pydantic, python-dotenv).
2. `train_baseline.py` resided in `scripts/`, contrary to the rule reserving `scripts/` strictly for thin CLI runners.
3. `tests/unit/test_baseline_acceptance.py` imported PyTorch and the legacy training script, causing the default unit test suite to require heavyweight ML libraries.
4. Development was performed without an isolated project virtual environment (`.venv`).

These conditions violated the lightweight bootstrap requirements for Checkpoint 00.

---

## Decision

1. **Lightweight Project Dependencies:**
   - Stripped all heavyweight ML and application dependencies from `dependencies = []` in `pyproject.toml`.
   - Preserved only lightweight development dependencies (`pytest>=7.4.0`, `ruff>=0.4.0`) under `[project.optional-dependencies] dev`.
   - Deferred ML dependency groups until the specific requirements of future implementation checkpoints (e.g., Checkpoint 01) are established.
2. **Isolation of Legacy Spike Artifacts:**
   - Relocated `scripts/train_baseline.py` to `experiments/legacy/train_baseline.py` via `git mv`.
   - Relocated `tests/unit/test_baseline_acceptance.py` to `experiments/legacy/test_baseline_acceptance.py` via `git mv`.
   - Preserved all code and git history without refactoring or deleting tests.
   - Documented in `experiments/legacy/README.md` that these files are exploratory legacy spikes excluded from the default test and CI suites.
3. **Dedicated Virtual Environment:**
   - Initialized `.venv` using Python 3.14.5.
   - Installed `finveritas` in editable mode (`pip install -e ".[dev]"`) strictly within `.venv`.
   - Verified that all linting (`ruff`), test runners (`pytest`), and package imports resolve to `.venv`.

---

## Consequences

- The standard bootstrap test suite (`pytest tests/`) executes in under 0.1s without requiring PyTorch, CUDA, or GPU drivers.
- Legacy spike code and acceptance tests remain fully intact and trackable in version control under `experiments/legacy/`.
- The repository structure adheres strictly to the documented ownership guidelines.
