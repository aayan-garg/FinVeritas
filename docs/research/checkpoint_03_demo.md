# FinVeritas — Checkpoint 03: Reproducibility & Working Demo

- **Date:** 2026-10-18
- **Checkpoint:** Checkpoint 03 (Reproducibility and Working Demo)
- **Status:** Complete, Verified, CPU-Only
- **Branch:** `checkpoint/03-reproducible-demo`

---

## 1. Overview & Objectives

Checkpoint 03 establishes end-to-end reproducibility and delivers a lightweight local inference CLI for question-evasion detection without requiring GPUs, network access, or neural transformer downloads.

Key achievements:
1. **Experimental Result Preservation:** Version-controlled the exact derived artifacts from Checkpoint 02 (`metrics.json`, `config.json`, `confusion_matrix.json`) under a strictly narrowed `.gitignore` rule.
2. **Scientific Governance & Disclaimers:** Clarified all documentation to reflect agreement with Eva-4B-V2 pseudo-labels, zero normalized-question overlap (with company/temporal leakage acknowledged as unruled-out), and explicit non-comparison to the unreleased 1K human gold set.
3. **Local Inference CLI:** Implemented `scripts/predict_evasion.py` with explicit `--train` and `--question`/`--answer` prediction flags.
4. **Zero-Warning Test Suite:** All 20 tests pass cleanly with zero warnings under Python 3.14.

---

## 2. Preserved Baseline Experimental Results

The held-out test evaluation results ($N = 3,346$, Fold 0) are saved under `artifacts/initial_baseline/`:

| Metric | Majority Class Baseline | TF-IDF + Logistic Regression |
| :--- | :--- | :--- |
| **Accuracy** | **52.30%** (0.5230) | **64.20%** (0.6420) |
| **Macro-F1** | **0.2289** | **0.5759** |
| **Weighted-F1** | **0.3592** | **0.6430** |
| **Fully Evasive F1** | 0.0000 | **0.4302** |
| **Fully Evasive Recall** | 0.0000 | **45.97%** (57 / 124) |
| **Train Time (CPU)** | 0.001 s | **36.83 s** |
| **Test Examples** | 3,346 | 3,346 |

### Version-Controlled Artifacts
- [`artifacts/initial_baseline/metrics.json`](file:///c:/Users/Aayan/Desktop/Project/NLP/artifacts/initial_baseline/metrics.json)
- [`artifacts/initial_baseline/config.json`](file:///c:/Users/Aayan/Desktop/Project/NLP/artifacts/initial_baseline/config.json)
- [`artifacts/initial_baseline/confusion_matrix.json`](file:///c:/Users/Aayan/Desktop/Project/NLP/artifacts/initial_baseline/confusion_matrix.json)

*Note: Heavy model weights (`*.joblib`) and raw datasets remain strictly ignored by Git.*

---

## 3. Scientific Wording & Governance Compliance

In accordance with strict research integrity standards:
- **Pseudo-Labels:** All dataset labels (`eva4b_label`) originate from **Eva-4B-V2** model annotations (arXiv:2601.09142), not human gold ground truth.
- **Agreement Metric:** Reported evaluation measures **agreement with the Eva-4B-V2 pseudo-label distribution**, not independently verified real-world evasion classification accuracy.
- **Leakage Boundary:** Splitting guarantees **zero normalized-question overlap between train and test** ($Train \cap Test = \emptyset$). However, company-level, speaker-level, and temporal leakage have **not been ruled out** due to the absence of metadata in the public dataset.
- **Gold Benchmark:** The 1,000-sample human-validated gold evaluation set from the EvasionBench paper is unreleased and unavailable. We do not claim state-of-the-art performance or compare against the paper's gold evaluation figures.
- **Uncalibrated Scores:** Model probability outputs represent softmax transformed decision scores from a linear classifier and are not calibrated Bayesian posterior probabilities.

---

## 4. Local Inference CLI Instructions

### Step 1: Fit and Save Model Artifact (One-Time Execution)
To train the baseline model on the local dataset (approx. 37s on CPU) and save the pipeline artifact:

```bash
python scripts/predict_evasion.py --train
```

This generates `artifacts/initial_baseline/model.joblib`. Inference subsequently loads this artifact directly without retraining.

### Step 2: Run Local Prediction
Run predictions on arbitrary question-answer pairs:

#### Example 1: Direct Financial Answer
```bash
python scripts/predict_evasion.py \
  --question "What was Q3 organic revenue growth in North America?" \
  --answer "Organic revenue grew 4.2% across North America, driven by strong enterprise renewals."
```

#### Example 2: Fully Evasive Answer
```bash
python scripts/predict_evasion.py \
  --question "Are you planning to lay off staff in the upcoming restructuring?" \
  --answer "We are always evaluating opportunities to optimize our cost structure and maximize shareholder value over the long horizon."
```

#### Example 3: Structured JSON Output
```bash
python scripts/predict_evasion.py \
  --question "Can you provide fiscal year 2027 gross margin guidance?" \
  --answer "We will provide 2027 guidance during our fourth quarter call later this year." \
  --json
```

---

## 5. Verification & Test Suite

Run quality gates:
```bash
# Test suite (20 tests, zero warnings)
python -m pytest -q

# Linter compliance
python -m ruff check .

# Dependency integrity
python -m pip check

# Trailing whitespace and merge conflict markers
git diff --check
```

### Warning Investigation & Resolution
In Checkpoint 02, a `UserWarning` (`The least populated class in y has only 4 members`) was observed during `StratifiedGroupKFold` split testing when synthetic toy fixtures had fewer members than the requested number of folds ($k=5$). In this checkpoint, the synthetic test fixture was updated to include 5 samples per class, resolving the warning completely. All 20 tests now pass with 0 warnings.
