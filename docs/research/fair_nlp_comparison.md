# FinVeritas: Fair NLP Benchmark, Truncation Audit, and Longer-Context Ablation

- **Date:** 2026-10-18
- **Checkpoint:** Checkpoint 05A (`checkpoint/05-nlp-fair-benchmark`)
- **Status:** Complete, Measured & Verified (NVIDIA RTX 3050 CUDA Accelerated)
- **Author:** FinVeritas Engineering Team
- **Base Commit:** `d9c3d61` (`checkpoint/04-transformer-evasion-nlp`)

---

## Executive Summary & Core Verdict

This document presents the rigorous experimental results for **Checkpoint 05A**, addressing the methodological asymmetry in prior baselines, conducting the first detailed tokenizer sequence-truncation audit, evaluating a longer-context 384-token neural model, auditing answer-level data leakage, and establishing 95% bootstrap confidence intervals for model differences.

### ⚠️ Essential Methodological & Data Governance Disclaimers
1. **Pseudo-Label Target:** All target labels (`eva4b_label`) in this dataset are **model-generated pseudo-labels** produced by **Eva-4B-V2** (arXiv:2601.09142), not independently verified human ground truth.
2. **Agreement Metric:** All metrics reported below measure empirical **agreement with the upstream Eva-4B-V2 pseudo-label distribution**, rather than independently verified real-world evasion-detection capability.
3. **Leakage Boundaries:** The grouping protocol guarantees **zero normalized-question overlap** across train, validation, and test partitions ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$). However, company-level, speaker-level, and temporal leakage have not been ruled out because the public dataset omits ticker, company, transcript, and call-date identifiers.
4. **Private Gold Benchmark:** The 1,000-sample human gold evaluation set reported in arXiv:2601.09142 is private and unreleased; it cannot be accessed or compared against in this local experiment.

---

## 1. Primary Empirical Findings

1. **Fair Baseline Comparison (10,704 Training Samples):**
   - The original TF-IDF baseline was trained on 13,380 samples (achieving 0.5759 Macro-F1).
   - When restricted to the identical 10,704 training examples available to the neural models, the **Fair TF-IDF baseline** achieves **0.5686 Macro-F1** and **0.4017 Fully-Evasive F1**.
   - Comparing against the fair baseline, `microsoft/deberta-v3-xsmall` (256 tokens) delivers an absolute gain of **+0.0255 Macro-F1** and **+0.0817 Fully-Evasive F1 (+20.34% relative)**.

2. **Longer-Context Ablation (384 Tokens):**
   - Increasing the sequence length budget from 256 to 384 tokens cuts test pair truncation from **45.13% down to 20.47%** (a **54.6% reduction in truncated pairs**).
   - The 384-token DeBERTa model achieves **0.5959 Macro-F1** and **0.5065 Fully-Evasive F1** (crossing the 0.50 F1 threshold on minority evasion).

3. **Bootstrap Significance Testing (1,000 Paired Resamples, Seed 42):**
   - **DeBERTa-256 vs Fair TF-IDF:** Fully-Evasive F1 delta mean is **+0.0824** with 95% CI `[+0.0022, +0.1688]` ($p = \mathbf{0.023}$). The neural improvement on minority evasion detection is statistically significant at $\alpha = 0.05$.
   - **DeBERTa-384 vs Fair TF-IDF:** Fully-Evasive F1 delta mean is **+0.1054** with 95% CI `[+0.0256, +0.1850]` ($p = \mathbf{0.004}$). Highly statistically significant.
   - **DeBERTa-384 vs DeBERTa-256:** Fully-Evasive F1 delta mean is **+0.0230** with 95% CI `[-0.0217, +0.0687]` ($p = 0.155$). While 384 tokens reduces truncation by more than half and edges F1 higher, the difference between the two context lengths remains within sampling noise on this test partition.

---

## 2. Side-by-Side Model Comparison Table

*All models evaluated on the identical 3,346 held-out test rows ($Train \cap Test = \emptyset$).*

| Metric | TF-IDF (Original 13k) | TF-IDF (Fair 10k) | DeBERTa-v3 (256) | DeBERTa-v3 (384) | Delta (384 vs Fair TF-IDF) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 64.20% | **64.49%** | 64.38% | 63.87% | -0.62% |
| **Macro-F1** | 0.5759 | 0.5686 | 0.5941 | **0.5959** | **+0.0273 (+4.80%)** |
| **Weighted-F1** | 0.6430 | **0.6451** | 0.6412 | 0.6328 | -0.0123 |
| **F1 (`direct`)** | 0.6777 | **0.6796** | 0.6241 | 0.6069 | -0.0727 |
| **F1 (`intermediate`)** | 0.6197 | 0.6245 | **0.6748** | 0.6742 | **+0.0497 (+7.96%)** |
| **F1 (`fully_evasive`)**| 0.4302 | 0.4017 | 0.4834 | **0.5065** | **+0.1048 (+26.09%)** |
| **Recall (`fully_evasive`)** | 45.97% | 38.71% | **64.52%** | 62.90% | **+24.19% (78 vs 48 caught)**|
| **Precision (`fully_evasive`)** | 40.43% | 41.74% | 38.65% | **0.4239** | **+0.65%** |
| **Test Pair Truncation** | 0.0% (N/A) | 0.0% (N/A) | 45.13% | **20.47%** | **-24.66% (-54.6% rel.)** |
| **Train Wall Time** | 36.8 s | **30.0 s** | 550.2 s | 655.9 s | 21.9x slower |
| **Eval Wall Time** | 3.7 s | **6.3 s** | 33.8 s | 53.3 s | 8.5x slower |
| **Peak GPU VRAM** | 0.0 MB | 0.0 MB | **686.4 MB** | 1,867.5 MB | Well within 6GB budget |

---

## 3. Detailed Per-Class Breakdown & Confusion Matrices

### 3.1 Per-Class Performance Breakdown

| Model | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF (Fair 10k)** | `direct` | 0.6898 | 0.6697 | **0.6796** | 1,750 |
| | `intermediate` | **0.6123** | 0.6372 | 0.6245 | 1,472 |
| | `fully_evasive` | 0.4174 | 0.3871 | 0.4017 | 124 |
| **DeBERTa-v3 (256)** | `direct` | 0.8089 | 0.5080 | 0.6241 | 1,750 |
| | `intermediate` | 0.5809 | 0.8050 | **0.6748** | 1,472 |
| | `fully_evasive` | 0.3865 | **0.6452** | 0.4834 | 124 |
| **DeBERTa-v3 (384)** | `direct` | **0.8251** | 0.4800 | 0.6069 | 1,750 |
| | `intermediate` | 0.5686 | **0.8281** | 0.6742 | 1,472 |
| | `fully_evasive` | **0.4239** | 0.6290 | **0.5065** | 124 |

---

### 3.2 Confusion Matrices

#### TF-IDF Fair Baseline (10,704 Train Samples)
```
Predicted --->       direct    intermediate    fully_evasive    Total
True:
direct               1,172         546              32          1,750
intermediate           499         938              35          1,472
fully_evasive           28          48              48            124
Total Predicted:     1,699       1,532             115          3,346
```

#### DeBERTa-v3-xsmall (256 Tokens)
```
Predicted --->       direct    intermediate    fully_evasive    Total
True:
direct                 889         817              44          1,750
intermediate           204       1,185              83          1,472
fully_evasive            6          38              80            124
Total Predicted:     1,099       2,040             207          3,346
```

#### DeBERTa-v3-xsmall (384 Tokens)
```
Predicted --->       direct    intermediate    fully_evasive    Total
True:
direct                 840         884              26          1,750
intermediate           173       1,219              80          1,472
fully_evasive            5          41              78            124
Total Predicted:     1,018       2,144             184          3,346
```

#### Confusion Analysis
1. **Catastrophic Evasion Misclassifications:** When an executive was `fully_evasive`, the fair TF-IDF baseline misclassified them as `direct` 28 times. DeBERTa-256 reduced this to 6 instances, and DeBERTa-384 reduced it to **only 5 instances** (an **82.1% reduction**).
2. **Intermediate Shift:** Both neural models exhibit strong sensitivity to executive hedging, classifying over 2,000 test cases as `intermediate`. This causes `direct` precision to reach **82.51%** for DeBERTa-384, but depresses `direct` recall as subtle hedging pushes answers into `intermediate`.

---

## 4. Sequence Truncation Mechanics Audit

### 4.1 Tokenizer Truncation Mechanism
Under HuggingFace Transformers, sequence-pair tokenization (`tokenizer(question, answer, truncation=True, max_length=M)`) utilizes the **`longest_first`** truncation strategy:
- Tokens are iteratively removed from whichever sequence is currently longer until the total sequence length (including special tokens `[CLS]`, `[SEP]`, `[SEP]`) equals `max_length`.
- We directly measured token retention using `tokenizer.sequence_ids()` to identify whether the analyst question, executive answer, or both lost tokens.

### 4.2 Empirical Sequence Length Distributions

| Partition | Sequence | Min | P25 | Median | P75 | P90 | P95 | P99 | Max | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train (10,704)** | Question | 4 | 46 | 68 | 96 | 127 | 148 | 200 | 574 | 74.5 |
| | Answer | 3 | 95 | 164 | 265 | 397 | 501 | 721 | 1,452 | 201.0 |
| | Total Pair | 13 | 158 | 242 | 357 | 501 | 613 | 857 | 1,537 | 278.5 |
| **Val (2,676)** | Question | 4 | 45 | 68 | 97 | 128 | 152 | 202 | 346 | 75.2 |
| | Answer | 3 | 92 | 164 | 272 | 409 | 510 | 739 | 1,697 | 202.7 |
| | Total Pair | 17 | 153 | 243 | 364 | 524 | 618 | 890 | 1,872 | 280.9 |
| **Test (3,346)** | Question | 8 | 46 | 67 | 95 | 125 | 151 | 202 | 324 | 74.3 |
| | Answer | 3 | 91 | 159 | 260 | 392 | 486 | 745 | 1,286 | 196.8 |
| | Total Pair | 17 | 154 | 237 | 352 | 497 | 598 | 858 | 1,370 | 274.1 |

### 4.3 Measured Truncation Rates: 256 vs. 384 Tokens

| Partition | Max Length | Pair Truncated | Answer Truncated | Question Truncated | Both Truncated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 256 tokens | 46.39% (4,966) | 46.11% (4,936) | 9.18% (983) | 8.90% (953) |
| | 384 tokens | **20.96% (2,244)**| **20.93% (2,240)**| **1.23% (132)** | **1.20% (128)** |
| **Val** | 256 tokens | 46.23% (1,237) | 45.78% (1,225) | 9.34% (250) | 8.89% (238) |
| | 384 tokens | **21.86% (585)** | **21.75% (582)** | **1.27% (34)** | **1.16% (31)** |
| **Test** | 256 tokens | 45.13% (1,510) | 44.80% (1,499) | 9.09% (304) | 8.76% (293) |
| | 384 tokens | **20.47% (685)** | **20.44% (684)** | **1.20% (40)** | **1.17% (39)** |

#### Truncation Audit Insights
1. **Answer Dominance:** In over 98% of truncated pairs, the executive answer is the sequence being truncated. Because median answer length is 159–164 tokens and 90th percentile is ~400 tokens, answers bear nearly all truncation burden.
2. **Question Loss Mitigation:** At 256 tokens, 9.09% of questions lost trailing tokens when questions were unusually verbose (e.g. multi-part analyst inquiries). At 384 tokens, question truncation dropped to **1.20%**, effectively eliminating question truncation.

---

## 5. Normalized Answer Leakage Audit

While question-level leakage is guaranteed to be strictly zero by `StratifiedGroupKFold`, we audited whether identical normalized answer strings appear across partitions:

- **Unique Normalized Answers:**
  - Train: 10,578 (out of 10,704 rows; 98.82% unique)
  - Val: 2,654 (out of 2,676 rows; 99.18% unique)
  - Test: 3,316 (out of 3,346 rows; 99.10% unique)
- **Partition Intersections:**
  - $\text{Train} \cap \text{Val}$: 34 unique overlapping answers
  - $\text{Train} \cap \text{Test}$: **32 unique overlapping answers** (appearing in 35 test rows)
  - $\text{Val} \cap \text{Test}$: 13 unique overlapping answers
- **Test Sample Exposure:** Only **1.05%** (35 of 3,346) of test rows contain an answer that appeared in the training set.
- **Label Agreement on Duplicate Answers:** **97.14%** (34 of 35 instances shared the exact same Eva-4B-V2 label).
- **Nature of Overlapping Answers:** All overlapping instances are standard formulaic corporate phrases:
  - `"yes."` / `"no."` (labeled `direct`)
  - `"thank you."` (labeled `intermediate`)
  - `"we do not provide that guidance at this time."` (labeled `fully_evasive`)
- **Metadata Leakage Bound:** Company-level, speaker-level, and temporal leakage **cannot be ruled out** because public EvasionBench omits ticker, executive name, transcript ID, and date metadata.

---

## 6. Bootstrap Significance & Uncertainty Analysis

*1,000 paired bootstrap resamples on the 3,346 held-out test rows (fixed seed 42).*

### 6.1 Individual Model Bootstrap 95% Confidence Intervals

| Model | Metric | Mean | Std Dev | 95% Bootstrap CI |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF (Fair 10k)** | Macro-F1 | 0.5676 | 0.0153 | `[0.5383, 0.5973]` |
| | Fully-Evasive F1 | 0.4001 | 0.0414 | `[0.3214, 0.4777]` |
| **DeBERTa-v3 (256)** | Macro-F1 | 0.5934 | 0.0133 | `[0.5681, 0.6197]` |
| | Fully-Evasive F1 | 0.4824 | 0.0340 | `[0.4140, 0.5519]` |
| **DeBERTa-v3 (384)** | Macro-F1 | 0.5952 | 0.0131 | `[0.5677, 0.6199]` |
| | Fully-Evasive F1 | 0.5055 | 0.0341 | `[0.4350, 0.5698]` |

---

### 6.2 Pairwise Differences and Empirical Hypothesis Testing

#### 1. DeBERTa-256 vs. Fair TF-IDF Baseline
- **Fully-Evasive F1 Delta:** Mean = **+0.0824**, 95% CI = `[+0.0022, +0.1688]`, $p = \mathbf{0.023}$
  - *Verdict:* Zero is outside the 95% confidence interval. DeBERTa-256 demonstrates a **statistically significant advantage** over the fair baseline in detecting evasive responses ($p < 0.05$).
- **Macro-F1 Delta:** Mean = **+0.0258**, 95% CI = `[-0.0047, +0.0564]`, $p = 0.051$
  - *Verdict:* Marginally significant at the 95% confidence level.

#### 2. DeBERTa-384 vs. Fair TF-IDF Baseline
- **Fully-Evasive F1 Delta:** Mean = **+0.1054**, 95% CI = `[+0.0256, +0.1850]`, $p = \mathbf{0.004}$
  - *Verdict:* **Highly statistically significant** ($p < 0.01$). The 95% confidence interval is strictly positive and bounded away from zero.
- **Macro-F1 Delta:** Mean = **+0.0276**, 95% CI = `[-0.0038, +0.0592]`, $p = 0.050$
  - *Verdict:* Borderline statistically significant overall macro gain.

#### 3. DeBERTa-384 vs. DeBERTa-256 (Context Ablation)
- **Fully-Evasive F1 Delta:** Mean = **+0.0230**, 95% CI = `[-0.0217, +0.0687]`, $p = 0.155$
- **Macro-F1 Delta:** Mean = **+0.0018**, 95% CI = `[-0.0176, +0.0215]`, $p = 0.433$
- *Verdict:* Expanding the context window from 256 to 384 tokens cuts answer truncation from 44.8% to 20.4% and yields positive empirical point estimates (+0.023 on evasive F1), but the difference between 384 and 256 is **not statistically distinguishable from zero** under bootstrap resampling ($p = 0.155$).

---

## 7. Deliverable Verification & Checkpoint Integrity

All artifacts and tests for Checkpoint 05A are verified:
- Benchmark Orchestrator: [`scripts/run_fair_benchmark.py`](file:///c:/Users/Aayan/Desktop/Project/NLP/.worktrees/transformer-nlp/scripts/run_fair_benchmark.py)
- Core Logic & Audits: [`src/finveritas/nlp_evasion/fair_benchmark.py`](file:///c:/Users/Aayan/Desktop/Project/NLP/.worktrees/transformer-nlp/src/finveritas/nlp_evasion/fair_benchmark.py)
- Research Documentation: [`docs/research/fair_nlp_comparison.md`](file:///c:/Users/Aayan/Desktop/Project/NLP/.worktrees/transformer-nlp/docs/research/fair_nlp_comparison.md)
- Serialized Metrics: [`artifacts/nlp_comparison/metrics.json`](file:///c:/Users/Aayan/Desktop/Project/NLP/.worktrees/transformer-nlp/artifacts/nlp_comparison/metrics.json)
- Serialized Config: [`artifacts/nlp_comparison/config.json`](file:///c:/Users/Aayan/Desktop/Project/NLP/.worktrees/transformer-nlp/artifacts/nlp_comparison/config.json)
- Unit Test Suite: [`tests/unit/test_nlp_comparison.py`](file:///c:/Users/Aayan/Desktop/Project/NLP/.worktrees/transformer-nlp/tests/unit/test_nlp_comparison.py) (5/5 passed)
- Git Hygiene: Heavy weights (`*.pt`, `checkpoints_384/`) are strictly Git-ignored.
