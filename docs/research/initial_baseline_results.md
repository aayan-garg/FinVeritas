# FinVeritas: Initial NLP Evasion Baseline Results

- **Date:** 2026-10-18
- **Checkpoint:** Checkpoint 02 (Fast, Reproducible NLP Baselines)
- **Status:** Complete, Measured & Verified (CPU-Only)
- **Author:** FinVeritas Engineering Team

---

## Executive Summary & Core Verdict

This document presents the first empirical, reproducible experimental results for question-evasion detection in FinVeritas. Two CPU-only baseline models were implemented, trained, and evaluated on a strictly leakage-aware, question-grouped test split of the approved EvasionBench dataset (`FutureMa/EvasionBench`).

### ⚠️ Essential Methodological & Data Governance Disclaimers
1. **Pseudo-Label Nature:** All target labels (`eva4b_label`) in this dataset are **model-generated pseudo-labels** produced by **Eva-4B-V2** (arXiv:2601.09142), not independently verified human ground truth.
2. **Evaluation Scope:** All metrics reported below measure **agreement with the upstream Eva-4B-V2 pseudo-label distribution**, rather than independently verified real-world evasion-detection performance.
3. **Leakage Boundaries:** The grouping protocol guarantees **zero normalized-question overlap between train and test**. However, **company-level, speaker-level, and temporal leakage have not been ruled out** because the public dataset omits ticker, company, transcript, and call-date identifiers.
4. **Gold Benchmark Unavailability:** The 1,000-sample human gold evaluation set reported in arXiv:2601.09142 is private and unreleased; it cannot be accessed or compared against in this local experiment. These scores must not be claimed as state-of-the-art or compared directly against human gold metrics.

### Key Experimental Findings
1. **Majority-Class Baseline:** Predicts `direct` for all test samples, achieving **52.30% Accuracy** and **0.2289 Macro-F1** (collapsing completely on `intermediate` and `fully_evasive`).
2. **TF-IDF + Balanced Logistic Regression Baseline:** Combines word n-grams (1–2) and character n-grams (3–5) with class-weighted logistic regression, achieving **64.20% Accuracy**, **0.5759 Macro-F1**, and **0.6430 Weighted-F1**.
3. **Minority-Class Defense:** Balanced class weighting successfully prevents the severe 3.7% minority class (`fully_evasive`) from being suppressed, yielding **45.97% Recall** and **0.4302 F1** on evasive responses without neural model fine-tuning.
4. **Primary Failure Mode:** Linguistic confusion between `direct` and `intermediate` constitutes 87.5% of all misclassifications (550 direct answers misclassified as intermediate; 497 intermediate answers misclassified as direct), reflecting executive hedging and partial answers.

---

## 1. Experimental Setup & Leakage-Aware Splitting

### 1.1. Dataset Source
- **Physical File:** `data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet`
- **Total Samples:** `16,726`
- **SHA-256:** `8ca4194c00cb037e9d719d45a569935aaabb31c3f781f03c1fcb9c7db875508b`
- **Target Field:** `eva4b_label`
- **Allowed Feature Inputs:** `question`, `answer`
- **Prohibited Fields (Excluded from Features):** `choices`, `eva4b_label_letter`, `uid`

### 1.2. Question-Grouped Stratified Split Protocol
To prevent data contamination where the same analyst question appears in both training and test sets, a group-aware splitting protocol was enforced:
- **Normalization:** Unicode NFKC normalization, lowercasing, whitespace collapsing and stripping applied to question text.
- **Unique Question Groups:** `16,610` unique normalized question groups across 16,726 rows.
- **Splitting Algorithm:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- **Evaluation Fold:** Fold `0` reserved as the held-out test set (exactly 20.00% of samples).
- **Group Overlap Verification:** Confirmed **zero normalized-question overlap between train and test** ($Train \cap Test = \emptyset$).
- **Preprocessing Isolation:** All TF-IDF vocabularies, IDF weights, and feature transforms were fitted strictly on the training partition.

### 1.3. Split Distribution Table

| Partition | Total Rows | Direct (Count / %) | Intermediate (Count / %) | Fully Evasive (Count / %) | Question Groups |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 13,380 (80.00%) | 6,999 (52.31%) | 5,887 (44.00%) | 494 (3.69%) | 13,288 |
| **Test Set** | 3,346 (20.00%) | 1,750 (52.30%) | 1,472 (43.99%) | 124 (3.71%) | 3,322 |
| **Full Corpus** | 16,726 (100.0%) | 8,749 (52.31%) | 7,359 (44.00%) | 618 (3.69%) | 16,610 |

---

## 2. Models & Architectures

### Experiment A: Majority-Class Minimal Baseline
- **Strategy:** Identifies the empirical mode of the training labels (`direct`, 52.31%) and outputs `direct` unconditionally for every test sample.
- **Purpose:** Establishes the floor performance metric that any meaningful NLP model must exceed.

### Experiment B: Word + Character TF-IDF with Balanced Logistic Regression
- **Feature Pipeline:**
  - Input text formatted as: `Question: {question}\nAnswer: {answer}`
  - **Word TF-IDF:** n-gram range `(1, 2)`, `min_df=3`, `max_features=40,000`, sublinear term frequency scaling enabled.
  - **Character TF-IDF:** n-gram range `(3, 5)`, `analyzer="char_wb"`, `min_df=5`, `max_features=60,000`, sublinear term frequency scaling enabled.
  - **FeatureUnion:** Concatenates word and character feature spaces into a single high-dimensional sparse matrix.
- **Classifier:**
  - `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs")`
  - Inversely weights loss per class to penalize misclassifications on the 3.7% `fully_evasive` minority class.

---

## 3. Measured Experimental Results

Both models were evaluated on the identical 3,346 held-out test rows.

### 3.1. Overall Performance Summary

| Metric | Experiment A: Majority Class | Experiment B: TF-IDF + Logistic Regression | Absolute Gain |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **52.30%** | **64.20%** | **+11.90%** |
| **Macro-F1** | **0.2289** | **0.5759** | **+0.3470** |
| **Weighted-F1** | **0.3592** | **0.6430** | **+0.2838** |
| **Train Time** | 0.001 s | 36.83 s | — |
| **Eval Time** | 0.000 s | 3.74 s | — |
| **Compute Device** | CPU | CPU | — |

### 3.2. Detailed Per-Class Metrics

#### Experiment A: Majority-Class Baseline
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **`direct`** | 0.5230 | 1.0000 | 0.6868 | 1,750 |
| **`intermediate`** | 0.0000 | 0.0000 | 0.0000 | 1,472 |
| **`fully_evasive`** | 0.0000 | 0.0000 | 0.0000 | 124 |
| **Macro Average** | **0.1743** | **0.3333** | **0.2289** | 3,346 |
| **Weighted Average** | **0.2735** | **0.5230** | **0.3592** | 3,346 |

#### Experiment B: TF-IDF + Balanced Logistic Regression
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **`direct`** | 0.6914 | 0.6646 | 0.6777 | 1,750 |
| **`intermediate`** | 0.6093 | 0.6304 | 0.6197 | 1,472 |
| **`fully_evasive`** | 0.4043 | 0.4597 | 0.4302 | 124 |
| **Macro Average** | **0.5683** | **0.5849** | **0.5759** | 3,346 |
| **Weighted Average** | **0.6447** | **0.6420** | **0.6430** | 3,346 |

---

## 4. Confusion Matrix Analysis

### Confusion Matrix (Test Set, N = 3,346)

```text
                     Predicted: direct    Predicted: intermediate    Predicted: fully_evasive    Total True
True: direct              1,163 (66.5%)               550 (31.4%)                  37 ( 2.1%)         1,750
True: intermediate          497 (33.8%)               928 (63.0%)                  47 ( 3.2%)         1,472
True: fully_evasive          22 (17.7%)                45 (36.3%)                  57 (46.0%)           124
-----------------------------------------------------------------------------------------------------------
Total Predicted           1,682                     1,523                         141                 3,346
```

### Key Diagnostic Observations
1. **Direct vs. Intermediate Ambiguity:**
   - 550 actual `direct` responses were predicted as `intermediate`.
   - 497 actual `intermediate` responses were predicted as `direct`.
   - Combined, this boundary accounts for **1,047 of 1,198 total errors (87.4%)**. Bag-of-words and n-gram representations struggle to distinguish polite executive qualifications ("we believe", "we anticipate") that accompany direct numerical answers from genuine hedging where the core query is unanswered.
2. **Evasion Detection Efficacy:**
   - Despite representing only 3.71% of test instances (124 rows), `fully_evasive` was correctly identified in **57 instances (45.97% recall)** with **40.43% precision**.
   - 45 evasive answers were mistaken for `intermediate`, which is linguistically expected given that intermediate answers often share evasive framing.
   - Only 22 evasive answers were classified as `direct`.

---

## 5. Environment & Execution Provenance

- **Python Version:** `3.14.5`
- **Scikit-Learn Version:** `1.9.1`
- **NumPy Version:** `2.5.4`
- **PyArrow Version:** `26.0.0`
- **OS Platform:** `Windows-11-10.0.26300-SP0`
- **Hardware Profile:** AMD64 CPU, 16 GB System RAM, NVIDIA RTX 3050 Laptop GPU (6 GB VRAM)
- **GPU Usage:** **0% (CPU execution only; no CUDA or PyTorch utilized)**
- **Artifact Files Generated:**
  - `artifacts/initial_baseline/metrics.json`
  - `artifacts/initial_baseline/config.json`
  - `artifacts/initial_baseline/confusion_matrix.json`

---

## 6. Conclusion & Smallest Justified Next Step

### Conclusion
The leakage-aware TF-IDF baseline establishes a credible, reproducible performance floor of **64.20% Accuracy** and **0.5759 Macro-F1** on CPU in under 45 seconds of training. However, linear bag-of-words classifiers are inherently limited in their ability to model relational question-answer semantics and subtle discursive evasion tactics.

### Next Step: Checkpoint 03 (Neural Sequence Classifier)
The logical next engineering step is to fine-tune a lightweight cross-encoder transformer (such as **DeBERTa-v3-small**, ~44M parameters) within the strict 6GB VRAM budget:
- Feed `[CLS] question [SEP] answer [SEP]` to model direct interactive attention between question constraints and answer clauses.
- Retain the identical StratifiedGroupKFold test split (Fold 0) to allow direct, un-leaked comparison with these TF-IDF baseline metrics.
- Apply class-weighted cross-entropy loss matching the ~9:1 class weight ratio established in the audit.

---
*End of FinVeritas Initial Baseline Results.*
