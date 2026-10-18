# ADR-0001: FinVeritas Repository Bootstrap & Exploratory Script Inventory

## Context & Problem Statement
Prior to Checkpoint 00, initial exploratory research into EvasionBench produced three scripts at the repository root:
1. `dataset_inspector.py`: Direct Parquet metadata inspector and duplicate analyzer for the Hugging Face `FutureMa/EvasionBench` release.
2. `train_baseline.py`: Baseline classification training pipeline using DeBERTa-v3 with class-weighting and 6GB VRAM memory budgeting.
3. `acceptance_tests.py`: Unit tests validating label mappings, class weight imbalance math, and memory limits.

Leaving Python scripts scattered across the root violates package modularity, complicates testing, and creates import ambiguity.

---

## Decision

1. **Establish Standard `src/` Package Structure:**
   Adopt the architectural layout specified in `docs/architecture/repository_structure.md`.
2. **Inventory & Relocate Exploratory Files:**
   - **`dataset_inspector.py` $\to$ `scripts/dataset_inspector.py`:** Retained as a thin CLI utility for dataset provenance auditing.
   - **`train_baseline.py` $\to$ `scripts/train_baseline.py`:** Preserved under `scripts/` as an exploratory training script. When Checkpoint 01 (NLP Evasion) is officially authorized, reusable components (`compute_class_weights`, `evaluate_predictions`, `EvasionDataset`) will be migrated into `src/finveritas/nlp_evasion/` to adhere strictly to the rule prohibiting premature ML module population.
   - **`acceptance_tests.py` $\to$ `tests/unit/test_baseline_acceptance.py`:** Preserved under `tests/unit/` to continue verifying mathematical correctness of class weights and metric calculations.
3. **Configure Strict Data & Artifact Exclusions in `.gitignore`:**
   Ensure all raw/interim/processed datasets, checkpoints, caches, and large binary artifacts cannot accidentally enter version control.

---

## File Inventory Table

| Original File | Target Location | Category | Disposition & Rationale |
| :--- | :--- | :--- | :--- |
| `dataset_inspector.py` | `scripts/dataset_inspector.py` | CLI Utility | Preserved in `scripts/`. Thin utility to download and inspect HF parquet without business logic dependencies. |
| `train_baseline.py` | `scripts/train_baseline.py` | Exploratory Spike | Preserved in `scripts/`. Reusable ML logic will be refactored into `src/finveritas/nlp_evasion/` in Checkpoint 01. |
| `acceptance_tests.py` | `tests/unit/test_baseline_acceptance.py` | Automated Test | Preserved in `tests/unit/`. Updated imports to reference `scripts.train_baseline`. Validates class-weight logic. |

---

## Consequences

- The repository root is completely clean, containing only configuration and top-level documentation.
- Git history preserves the original root snapshot commit (`6e8f117`) and tracks renames cleanly.
- Modules in `src/finveritas/` remain clean skeletons until authorized.
