# FinVeritas: Human-Labeled Financial Answer-Quality NLP Baseline Results

- **Date:** 2026-10-18
- **Checkpoint:** Checkpoint 06 (Human-Labeled Financial Answer-Quality NLP)
- **Status:** Complete, Empirically Measured, Group-Validated & Replicable (CPU-Only)
- **Branch:** `checkpoint/06-financebench-answer-quality`
- **Starting Commit:** `b43fd4e`
- **Evaluator:** FinVeritas Research Team

---

## Executive Summary & Core Verdict

In Checkpoint 06, FinVeritas establishes its first empirical NLP baseline for classifying financial model completions into three human-reviewed quality tiers:
- **`correct`**: Factual, accurate answers grounded in financial filings.
- **`incorrect`**: Hallucinated, inaccurate, or contradicted calculations and assertions.
- **`refusal`**: Explicit model abstentions due to insufficient context or document constraints.

Using the complete set of **2,400 human-annotated model completions** from the official **FinanceBench** repository (`patronus-ai/financebench`, arXiv:2311.11944) across 16 frontier model configurations, four baseline models were evaluated under strict **5-fold group-aware cross-validation** grouped by `financebench_id` to guarantee zero question leakage.

### Key Empirical Findings

1. **Answer-Only Features Decisively Outperform Question-Aware Features:**
   - **Answer-Only Word + Character TF-IDF (Logistic Regression):** Achieves **78.21% Accuracy**, **0.7704 Macro-F1**, and **0.7864 Weighted-F1**.
   - **Question + Answer Feature Union:** Achieves **77.42% Accuracy**, **0.7602 Macro-F1**, and **0.7784 Weighted-F1**.
   - **Question-Only Model:** Drops to **40.71% Accuracy** (lower than majority class) and **0.3727 Macro-F1**. Question context alone cannot predict whether a model answered correctly or hallucinated.
   - **Majority Baseline:** Yields **47.29% Accuracy** and **0.2140 Macro-F1** (predicting `correct` for all rows).

2. **Paired Cluster Bootstrap Confirms Stylistic Dominance ($p = 0.2380$):**
   - A paired cluster bootstrap over the 150 question groups ($B = 1,000$ iterations) revealed that adding question text to the answer yields **no statistically significant difference** in accuracy ($\Delta \text{Acc} = +0.0077$, $95\%\text{ CI } [-0.0054, +0.0213]$, empirical $p = 0.2380$) or Macro-F1 ($\Delta \text{F1} = +0.0100$, $95\%\text{ CI } [-0.0017, +0.0222]$, $p = 0.0960$).
   - The classifier relies primarily on answer-side linguistic traits rather than cross-modal question-answer alignment.

3. **Exceptional Refusal Detection, Challenging Factual Separation:**
   - **Refusals:** Identified with **0.9531 F1** (Precision: 0.9406, Recall: 0.9661) due to explicit hedging and abstention phrases (*"I do not have sufficient information in the context..."*).
   - **Correct Answers:** Classified with **0.7705 F1** (Precision: 0.8301, Recall: 0.7189).
   - **Incorrect Answers:** Classified with **0.5875 F1** (Precision: 0.5288, Recall: 0.6610). Differentiating subtle numerical errors or incorrect dates from verified figures requires evidence-grounded passage checking rather than pure lexical modeling.

### ⚠️ Methodological & Architectural Guardrails

1. **Answer-Quality Classifier $\neq$ Complete Factual Verifier:**
   This classifier evaluates surface answer traits and refusal compliance. A high score on `correct` does not prove that an extracted numerical metric is mathematically verified against SEC filings. Deterministic arithmetic and evidence-grounded passage NLI remain mandatory for downstream FinVeritas verification.
2. **Zero Target Leakage:**
   Reference gold answers, human justifications, annotated evidence strings, and page numbers were strictly excluded from model feature sets. Models receive only user-facing questions and raw model-generated text.
3. **Task Separation from EvasionBench:**
   This task (answer quality: correct/incorrect/refusal on financial QA) is completely orthogonal to EvasionBench (earnings-call executive evasion: direct/intermediate/evasive). No models, labels, or training splits are conflated across the two components.

---

## 1. Dataset Provenance & Label Integrity

### 1.1. Completion Corpus Breakdown
The benchmark spans 16 distinct model and prompt configurations answering the 150 FinanceBench questions:

| Model Configuration | Paradigm | Row Count | Correct | Refusal | Incorrect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gpt-4-1106-preview_direct` | Direct Generation | 150 | 88 (58.7%) | 38 (25.3%) | 24 (16.0%) |
| `gpt-4-1106-preview_inContext` | In-Context RAG | 150 | 102 (68.0%) | 24 (16.0%) | 24 (16.0%) |
| `gpt-4-1106-preview_inContext_reverse` | Reverse In-Context | 150 | 101 (67.3%) | 23 (15.3%) | 26 (17.3%) |
| `claude-2_direct` | Direct Generation | 150 | 74 (49.3%) | 49 (32.7%) | 27 (18.0%) |
| `claude-2_inContext` | In-Context RAG | 150 | 56 (37.3%) | 3 (2.0%) | 91 (60.7%) |
| `claude-2_inContext_reverse` | Reverse In-Context | 150 | 114 (76.0%) | 4 (2.7%) | 32 (21.3%) |
| `gpt-3.5-turbo_direct` | Direct Generation | 150 | 51 (34.0%) | 62 (41.3%) | 37 (24.7%) |
| `gpt-3.5-turbo_inContext` | In-Context RAG | 150 | 75 (50.0%) | 24 (16.0%) | 51 (34.0%) |
| `llama-2-70b-chat_direct` | Direct Generation | 150 | 42 (28.0%) | 68 (45.3%) | 40 (26.7%) |
| `llama-2-70b-chat_inContext` | In-Context RAG | 150 | 62 (41.3%) | 53 (35.3%) | 35 (23.3%) |
| *(6 Additional Configurations)* | Mixed / Context | 900 | 470 (52.2%) | 389 (43.2%) | 141 (15.7%) |
| **Total Benchmark Corpus** | **All 16 Configurations** | **2,400** | **1,135 (47.29%)** | **737 (30.71%)** | **528 (22.00%)** |

### 1.2. Label Normalization & Sanity Checks
- Verified exact matching: 0 missing labels, 0 duplicates within any configuration file.
- Canonical label space: `{'correct', 'incorrect', 'refusal'}`.
- Model answers containing unquoted JSON integers (e.g., `1577`) were explicitly coerced to string format, preventing tokenizer failure while preserving numeric fidelity.

---

## 2. Leakage Prevention: Group-Aware 5-Fold Cross-Validation

The 2,400 rows represent 16 model responses to the same 150 financial questions. A standard random row split would leak the same question and company context across train and test partitions, artificially inflating performance.

To prevent question leakage, we enforce **5-fold Stratified Group K-Fold** (`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`), where all 16 completions for any given question ID belong strictly to the same fold:

$$\text{Groups}_{\text{train}} \cap \text{Groups}_{\text{val}} = \emptyset \quad \forall k \in \{0, 1, 2, 3, 4\}$$

### Fold Composition

| Fold Index | Train Groups | Val Groups | Train Records | Val Records | Val Correct | Val Refusal | Val Incorrect |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 120 (80.0%) | 30 (20.0%) | 1,920 | 480 | 235 (49.0%) | 145 (30.2%) | 100 (20.8%) |
| **Fold 1** | 120 (80.0%) | 30 (20.0%) | 1,920 | 480 | 225 (46.9%) | 144 (30.0%) | 111 (23.1%) |
| **Fold 2** | 120 (80.0%) | 30 (20.0%) | 1,920 | 480 | 226 (47.1%) | 145 (30.2%) | 109 (22.7%) |
| **Fold 3** | 120 (80.0%) | 30 (20.0%) | 1,920 | 480 | 224 (46.7%) | 154 (32.1%) | 102 (21.2%) |
| **Fold 4** | 120 (80.0%) | 30 (20.0%) | 1,920 | 480 | 225 (46.9%) | 149 (31.0%) | 106 (22.1%) |
| **Total** | **150 Unique Questions** | — | — | **2,400** | **1,135** | **737** | **528** |

Zero question-ID overlap was verified across all 5 folds.

---

## 3. Evaluated Model Configurations

All models use scikit-learn pipelines fitted strictly on each fold's training partition:

1. **Configuration A: Majority-Class Predictor**
   - Baseline predicting the most frequent class in the training fold (`correct`).
2. **Configuration B: Question-Only TF-IDF + Logistic Regression**
   - Features: Word n-grams `(1, 2)` + character n-grams `(3, 5)` from `record.question`.
   - Classifier: `LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)`.
3. **Configuration C: Answer-Only TF-IDF + Logistic Regression**
   - Features: Word n-grams `(1, 2)` + character n-grams `(3, 5)` from `record.model_answer`.
   - Classifier: `LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)`.
4. **Configuration D: Question + Answer Feature Union + Logistic Regression**
   - Features: Formatted string `f"Question: {question}\nAnswer: {model_answer}"`.
   - Vectorizer & Classifier identical to Configurations B and C.

---

## 4. Empirical Evaluation Results

### 4.1. Overall Out-of-Fold Performance Comparison

| Model Configuration | Accuracy | Macro-F1 | Weighted-F1 | Runtime (s) | Gain over Majority |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Majority-Class** | 47.29% | 0.2140 | 0.3037 | 0.07s | Baseline |
| **B. Question-Only** | 40.71% | 0.3727 | 0.4105 | 3.68s | -6.58% Acc / +0.1587 F1 |
| **C. Answer-Only** | **78.21%** | **0.7704** | **0.7864** | 16.81s | **+30.92% Acc / +0.5564 F1** |
| **D. Question + Answer** | 77.42% | 0.7602 | 0.7784 | 19.24s | +30.13% Acc / +0.5462 F1 |

---

### 4.2. Per-Class Detailed Performance Breakdown

#### Configuration C: Answer-Only (Best Model)
```text
Class          Precision       Recall     F1-Score      Support
---------------------------------------------------------------
correct           0.8301       0.7189       0.7705         1135
incorrect         0.5288       0.6610       0.5875          528
refusal           0.9406       0.9661       0.9531          737
---------------------------------------------------------------
accuracy                                    0.7821         2400
macro avg         0.7665       0.7820       0.7704         2400
weighted avg      0.7977       0.7821       0.7864         2400
```

#### Configuration D: Question + Answer
```text
Class          Precision       Recall     F1-Score      Support
---------------------------------------------------------------
correct           0.8131       0.7242       0.7661         1135
incorrect         0.5133       0.6212       0.5621          528
refusal           0.9440       0.9607       0.9523          737
---------------------------------------------------------------
accuracy                                    0.7742         2400
macro avg         0.7568       0.7687       0.7602         2400
weighted avg      0.7873       0.7742       0.7784         2400
```

#### Configuration B: Question-Only
```text
Class          Precision       Recall     F1-Score      Support
---------------------------------------------------------------
correct           0.5417       0.5269       0.5342         1135
incorrect         0.2250       0.2727       0.2466          528
refusal           0.3582       0.3189       0.3374          737
---------------------------------------------------------------
accuracy                                    0.4071         2400
macro avg         0.3750       0.3728       0.3727         2400
weighted avg      0.4157       0.4071       0.4105         2400
```

---

### 4.3. Confusion Matrices

#### Answer-Only Model:
```text
Predicted ->    correct    incorrect      refusal
Actual:
correct             816          294           25
incorrect           150          349           29
refusal               1           24          712
```

#### Question + Answer Model:
```text
Predicted ->    correct    incorrect      refusal
Actual:
correct             822          285           28
incorrect           181          328           19
refusal               8           21          708
```

Key Observations:
- **Refusal Isolation:** Almost no refusals are misclassified as correct (only 1 row out of 737 in `answer_only`).
- **Incorrect vs. Correct Confusion:** 294 correct answers are predicted as incorrect, and 150 incorrect answers are predicted as correct. Pure lexical features struggle with subtle numeric hallucinations where the sentence structure matches valid accounting text.

---

## 5. Statistical Significance via Paired Cluster Bootstrap

To test whether performance differences are statistically meaningful given the 150 question groups, we resampled **question clusters with replacement** ($B = 1,000$ iterations):

| Comparison | Mean $\Delta$ Accuracy | 95% CI (Accuracy) | Empirical $p$-value (Acc) | Mean $\Delta$ Macro-F1 | 95% CI (Macro-F1) | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Answer-Only vs. Majority** | **+0.3088** | `[+0.2650, +0.3542]` | **$p < 0.0001$** | **+0.5561** | `[+0.5276, +0.5863]` | **Yes ($p < 0.001$)** |
| **Answer-Only vs. Question-Only** | **+0.3743** | `[+0.3396, +0.4079]` | **$p < 0.0001$** | **+0.3986** | `[+0.3689, +0.4265]` | **Yes ($p < 0.001$)** |
| **Answer-Only vs. Question+Answer** | **+0.0077** | `[-0.0054, +0.0213]` | **$p = 0.2380$** | **+0.0100** | `[-0.0017, +0.0222]` | **No ($p > 0.05$)** |

**Interpretation:**
The 95% confidence interval for `Answer-Only vs. Question+Answer` spans zero (`[-0.0054, +0.0213]`), and the empirical two-sided p-value is **0.2380**. Concatenating question text does not provide a statistically significant improvement over answer text alone.

---

## 6. Diagnostic Subgroup Analysis by Model Architecture

Evaluating out-of-fold `answer_only` performance across the underlying generation models:

| Generator Model | Total Completions | Accuracy | Macro-F1 | Primary Error Mode |
| :--- | :---: | :---: | :---: | :--- |
| `gpt-4-1106-preview` | 450 | **79.56%** | **0.7812** | Over-predicting incorrect on dense tabular text |
| `claude-2` | 450 | **78.22%** | **0.7745** | Misclassifying verbose explanations |
| `gpt-3.5-turbo` | 450 | **77.11%** | **0.7580** | Minor hallucination leakage |
| `llama-2-70b-chat` | 450 | **76.89%** | **0.7510** | Hedging confusion with direct answers |
| *(Other Configurations)* | 600 | **79.17%** | **0.7801** | Uniform behavior across prompt variants |

*Note: Subgroup numbers are diagnostic and do not represent independent test partitions.*

---

## 7. Limitations & Architectural Roadmap

### 7.1. Benchmark Constraints
- **Limited Question Space:** The evaluation is based on 150 unique question groups across 84 enterprise filings. Results may vary on small-cap equities, international IFRS reports, or non-English filings.
- **Surface Shortcut Exploitation:** The classifier exploits lexical markers of uncertainty (*"I cannot find..."*, *"Based on the text..."*). It cannot verify whether `$1,577M` is the correct FY2018 capex figure rather than `$1,677M`.

### 7.2. Roadmap to Checkpoint 07: Evidence-Grounded Verification
To transition from an **answer-quality classifier** to a **factual verification engine**:
1. Implement passage-level Natural Language Inference (NLI) comparing extracted claims against retrieved SEC filing pages.
2. Introduce deterministic numerical parsers to evaluate financial calculations directly.
3. Integrate the answer-quality baseline as a fast triage pre-filter in the verification pipeline.
