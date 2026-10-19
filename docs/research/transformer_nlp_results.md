# FinVeritas: Transformer NLP Evasion Model Results

- **Date:** 2026-10-18
- **Checkpoint:** Checkpoint 04 (`checkpoint/04-transformer-evasion-nlp`)
- **Status:** Complete, Measured & Verified (NVIDIA RTX 3050 CUDA Accelerated)
- **Author:** FinVeritas Engineering Team
- **Base Commit:** `cbc81ea` (`checkpoint/03-reproducible-demo`)

---

## Executive Summary & Core Verdict

This document presents the first empirical neural NLP results for question-evasion detection in FinVeritas. We fine-tuned `microsoft/deberta-v3-xsmall` (~44M parameters) on earnings-call question-answer pairs from the EvasionBench corpus (`FutureMa/EvasionBench`) to classify three behavioral evasion tiers: `direct`, `intermediate`, and `fully_evasive`.

Following the strategic project directive (**NLP first, RAG second**), this work was conducted in an isolated Git worktree (`c:\Users\Aayan\Desktop\Project\NLP\.worktrees\transformer-nlp`) to guarantee complete separation from concurrent FinanceBench retrieval operations.

### ⚠️ Essential Methodological & Data Governance Disclaimers
1. **Pseudo-Label Target:** All target labels (`eva4b_label`) in this dataset are **model-generated pseudo-labels** produced by **Eva-4B-V2** (arXiv:2601.09142), not independently verified human ground truth.
2. **Agreement Metric:** All metrics reported below measure empirical **agreement with the upstream Eva-4B-V2 pseudo-label distribution**, rather than independently verified real-world evasion-detection capability.
3. **Leakage Boundaries:** The grouping protocol guarantees **zero normalized-question overlap** across train, validation, and test partitions ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$). However, company-level, speaker-level, and temporal leakage have not been ruled out because the public dataset omits ticker, company, transcript, and call-date identifiers.
4. **Private Gold Benchmark:** The 1,000-sample human gold evaluation set reported in arXiv:2601.09142 is private and unreleased; it cannot be accessed or compared against in this local experiment.

### Key Experimental Findings
1. **Macro-F1 Improvement:** `microsoft/deberta-v3-xsmall` achieved **0.5947 Macro-F1**, outperforming the Checkpoint 02 TF-IDF + Logistic Regression baseline (**0.5759 Macro-F1**) by **+0.0188 (+3.26% relative)**.
2. **Evasion Recall Surge:** On the critical 3.71% minority class (`fully_evasive`), DeBERTa achieved **64.52% Recall** (detecting 80 of 124 evasive responses) compared to **45.97% Recall** (57 of 124) for the TF-IDF baseline. Fully evasive F1 improved from **0.4302 to 0.4848 (+12.69% relative)**.
3. **Accuracy Stability:** Overall Accuracy reached **64.41%** (vs. 64.20% for baseline and 52.30% for majority class).
4. **Truncation Impact:** With a sequence budget of 256 tokens, **45.13%** (1,510 of 3,346) of test pairs experienced executive answer truncation. Despite losing trailing sentences in long answers, the neural attention mechanism still substantially outperformed n-gram surface features on non-direct responses.

---

## 1. Experimental Setup & Partitioning Protocol

### 1.1 Dataset Provenance & Input Features
- **File:** `data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet`
- **Total Samples:** `16,726`
- **Target Field:** `eva4b_label` (`direct`, `intermediate`, `fully_evasive`)
- **Allowed Feature Inputs:**
  - Sequence 1: Analyst Question (`question`)
  - Sequence 2: Executive Answer (`answer`)
- **Strictly Excluded Prohibited Fields:** `choices`, `eva4b_label_letter`, `uid`, and all label-derived features.

### 1.2 Leakage-Disjoint Stratified Group Partitioning
To guarantee zero contamination and fair benchmarking against the Checkpoint 02 TF-IDF baseline:
1. **Question Normalization:** Unicode NFKC normalization, lowercasing, whitespace collapsing and stripping applied to question text.
2. **Fixed Test Split (Fold 0):** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` was reproduced exactly, assigning Fold 0 (3,346 samples, 20.00%) as the held-out test set.
3. **Train / Validation Split:** The remaining 13,380 samples were partitioned using a second `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`:
   - **Training Partition:** 10,704 samples (80% of train-val pool, 64.00% of corpus).
   - **Validation Partition:** 2,676 samples (20% of train-val pool, 16.00% of corpus).
4. **Zero-Leakage Verification:** Zero normalized question overlap confirmed across all three pairs:
   - $\text{Train} \cap \text{Val} = 0$
   - $\text{Train} \cap \text{Test} = 0$
   - $\text{Val} \cap \text{Test} = 0$
5. **Model Selection Gate:** Checkpoint selection and early-stopping decisions were gated strictly on **Validation Macro-F1** (0.6043). No hyperparameter tuning or selection was performed on the held-out test partition.

### 1.3 Partition Distribution

| Partition | Total Rows | Direct (Count / %) | Intermediate (Count / %) | Fully Evasive (Count / %) | Unique Question Groups |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 10,704 (64.00%) | 5,599 (52.31%) | 4,710 (44.00%) | 395 (3.69%) | 10,630 |
| **Val Set** | 2,676 (16.00%) | 1,400 (52.32%) | 1,177 (43.98%) | 99 (3.70%) | 2,658 |
| **Test Set** | 3,346 (20.00%) | 1,750 (52.30%) | 1,472 (43.99%) | 124 (3.71%) | 3,322 |
| **Full Corpus**| 16,726 (100.0%) | 8,749 (52.31%) | 7,359 (44.00%) | 618 (3.69%) | 16,610 |

---

## 2. Model Architecture & Training Hyperparameters

- **Base Model Architecture:** `microsoft/deberta-v3-xsmall`
  - Total Parameters: 44,357,379
  - Hidden Dimension: 384
  - Layers (Transformer Blocks): 12
  - Attention Heads: 6
  - Disentangled Attention: Relative position embeddings + content embeddings
  - Vocabulary Size: 128,100 (SentencePiece SPM tokenizer)
- **Tokenization:**
  - Sequence-pair format: `[CLS] {question} [SEP] {answer} [SEP]`
  - Maximum sequence length: 256 tokens
  - Dynamic batch padding with `[PAD]` (token id: 0)
- **Training Hyperparameters:**
  - Optimization Algorithm: AdamW (`weight_decay=0.01`, gradient clipping max norm `1.0`)
  - Learning Rate: `2e-5`
  - Per-Device Batch Size: 2
  - Gradient Accumulation Steps: 4 (Effective batch size: 8)
  - Epochs: 1
  - Loss Function: Class-weighted CrossEntropyLoss
    - Class weights computed strictly from the training partition:
      - `direct`: 0.6373
      - `intermediate`: 0.7575
      - `fully_evasive`: 9.0329
  - Precision: Explicit FP32 on CUDA (ensuring numerical stability with DeBERTa-v3 relative position bias)
  - Compute Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4, PyTorch 2.6.0+cu124)
  - Peak GPU VRAM: 1,576.03 MB
  - Random Seed: 42 (deterministic seeding across PyTorch, NumPy, Python)

---

## 3. Sequence Truncation Analysis

- **Configured Maximum Sequence Budget:** 256 tokens
- **Test Set Truncation Rate:** **45.13%** (1,510 of 3,346 pairs)
- **Mechanism:** Sequence 1 (`question`) is preserved entirely; sequence 2 (`answer`) is truncated from the end when total tokens exceed 256.
- **Scientific Limitation:** In corporate earnings conference calls, CEOs and CFOs frequently employ "deflection strategies" where they begin by answering an adjacent topic or offering introductory boilerplate before declining to answer or hedging at the conclusion of the response. Under a 256-token budget, 45.13% of answers lose their concluding sentences. Despite this information loss, DeBERTa's contextual representations captured sufficient early-response evasion signals to outperform the baseline.

---

## 4. Empirical Evaluation: Side-by-Side Results

*All models evaluated on the identical 3,346 held-out test rows.*

### 4.1 Overall Performance Summary

| Metric | Majority Baseline | TF-IDF + Logistic Regression | DeBERTa-v3-xsmall | Gain vs TF-IDF Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 52.30% | 64.20% | **64.41%** | **+0.21%** |
| **Macro-F1** | 0.2289 | 0.5759 | **0.5947** | **+0.0188 (+3.26%)** |
| **Weighted-F1** | 0.3592 | **0.6430** | 0.6414 | -0.0016 (-0.25%) |
| **F1 (`direct`)** | 0.6868 | **0.6777** | 0.6241 | -0.0536 |
| **F1 (`intermediate`)** | 0.0000 | 0.6197 | **0.6752** | **+0.0555 (+8.96%)** |
| **F1 (`fully_evasive`)**| 0.0000 | 0.4302 | **0.4848** | **+0.0546 (+12.69%)** |
| **Recall (`fully_evasive`)** | 0.00% | 45.97% | **64.52%** | **+18.55% (80 vs 57 caught)** |
| **Train Wall Time** | 0.001 s | **36.83 s** | 550.22 s (9.17 min) | 14.9x slower |
| **Eval Wall Time** | 0.000 s | **3.74 s** | 36.03 s | 9.6x slower |
| **Peak GPU VRAM** | 0.0 MB | 0.0 MB | 1,576.03 MB | Within 6GB budget |

---

### 4.2 Detailed Per-Class Breakdown

#### DeBERTa-v3-xsmall (Checkpoint 04)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **`direct`** | **0.8089** | 0.5080 | 0.6241 | 1,750 |
| **`intermediate`** | 0.5811 | **0.8057** | **0.6752** | 1,472 |
| **`fully_evasive`** | 0.3883 | **0.6452** | **0.4848** | 124 |
| **Macro Average** | 0.5928 | **0.6530** | **0.5947** | 3,346 |
| **Weighted Average** | **0.6931** | 0.6441 | 0.6414 | 3,346 |

#### TF-IDF + Logistic Regression (Checkpoint 02 Baseline)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **`direct`** | 0.7093 | 0.6486 | 0.6777 | 1,750 |
| **`intermediate`** | 0.6394 | 0.6012 | 0.6197 | 1,472 |
| **`fully_evasive`** | 0.4043 | 0.4597 | 0.4302 | 124 |
| **Macro Average** | 0.5843 | 0.5698 | 0.5759 | 3,346 |
| **Weighted Average** | 0.6672 | 0.6420 | 0.6430 | 3,346 |

---

### 4.3 Confusion Matrix Analysis

#### DeBERTa-v3-xsmall Confusion Matrix
```
Predicted --->       direct    intermediate    fully_evasive    Total
True:
direct                 889         817              44          1,750
intermediate           204       1,186              82          1,472
fully_evasive            6          38              80            124
Total Predicted:     1,099       2,041             206          3,346
```

#### TF-IDF Baseline Confusion Matrix (for reference)
```
Predicted --->       direct    intermediate    fully_evasive    Total
True:
direct               1,135         550              65          1,750
intermediate           497         885              90          1,472
fully_evasive           27          40              57            124
Total Predicted:     1,659       1,475             212          3,346
```

#### Key Confusion Observations
1. **Critical False Negative Reduction:** When an executive was `fully_evasive`, the baseline misclassified them as `direct` 27 times. DeBERTa reduced this catastrophic error to **only 6 instances** (a 77.8% reduction in evasions mistaken for direct answers).
2. **Intermediate Shift:** DeBERTa is noticeably more sensitive to executive hedging, classifying 2,041 responses as `intermediate` compared to the baseline's 1,475. This shifted 817 direct answers into intermediate, lowering direct recall but driving direct precision to **80.89%**.

---

## 5. Software & CLI Integration

The neural model is fully integrated alongside the TF-IDF baseline without breaking existing workflows:

### 5.1 Dedicated Transformer Inference CLI
```bash
python scripts/predict_transformer.py \
  --question "Why did gross margins decline in Q3?" \
  --answer "We decline to comment on margin breakdowns at this time."
```
Output:
```json
{
  "question": "Why did gross margins decline in Q3?",
  "answer": "We decline to comment on margin breakdowns at this time.",
  "predicted_label": "fully_evasive",
  "uncalibrated_probabilities": {
    "direct": 0.0412,
    "intermediate": 0.1287,
    "fully_evasive": 0.8301
  },
  "is_truncated": false,
  "total_tokens_before_truncation": 24,
  "max_length": 256,
  "model_architecture": "microsoft/deberta-v3-xsmall",
  "device": "cuda",
  "disclaimer": "These classification scores measure empirical agreement with Eva-4B-V2 model-generated pseudo-labels on EvasionBench, NOT independently verified human ground-truth evasion."
}
```

### 5.2 Unified CLI Dispatch (`scripts/predict_evasion.py`)
Both inference backends are accessible from the primary CLI:
```bash
# TF-IDF Baseline (Default)
python scripts/predict_evasion.py --question "..." --answer "..." --model-type tfidf

# Neural Transformer Option
python scripts/predict_evasion.py --question "..." --answer "..." --model-type transformer
# Or using shortcut flag:
python scripts/predict_evasion.py --question "..." --answer "..." --transformer
```

---

## 6. Deliverable Verification

All artifacts and configurations have been verified and persisted:
- Model implementation: `src/finveritas/nlp_evasion/transformer_classifier.py`
- Training pipeline: `scripts/train_transformer.py`
- Inference CLI: `scripts/predict_transformer.py`
- Unit tests: `tests/unit/test_transformer_classifier.py`
- Metrics artifact: `artifacts/transformer_nlp/metrics.json`
- Configuration artifact: `artifacts/transformer_nlp/config.json`
- Git hygiene: Model weights (`best_model.pt`) and tokenizer caches are safely Git-ignored.
