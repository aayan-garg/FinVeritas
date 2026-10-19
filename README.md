# FinVeritas

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-70%20Passed%20(100%25)-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-black.svg)](https://github.com/astral-sh/ruff)

**FinVeritas** is an end-to-end financial intelligence and claim verification framework. It is engineered to detect managerial question-evasion during corporate earnings-call Q&A sessions and audit extracted financial claims against official regulatory filings (10-K, 10-Q, 8-K) through hybrid lexical-dense retrieval and evidence grounding.

---

## Executive Overview & Problem Statement

Corporate earnings calls are the primary real-time communication channel between public company leadership and capital markets. However, executives routinely employ rhetorical strategies—formulaic non-answers, selective omission, semantic hedging, and diversionary optimism—to evade challenging questions from research analysts.

Without systematic audit mechanisms:
1. **Analytic Asymmetry:** Financial analysts and algorithmic trading systems risk treating managerial deflections or ungrounded assertions as factual disclosures.
2. **Hallucination Risk:** Generative AI models applied to earnings transcripts frequently adopt the persuasive tone of executives without verifying numerical claims against audited SEC regulatory statements.

FinVeritas solves this problem through an audited, multi-stage architecture:
- **Discourse Evasion Detection:** Classifying dialogue turns into `direct`, `intermediate`, or `fully_evasive` responses using transformer classifiers calibrated with strict group-disjoint splits and paired bootstrap significance testing.
- **Shortcut Learning & Robustness Audits:** Quantifying answer-template memorization and evaluating models on strict answer-disjoint splits to guard against superficial corporate phrasing shortcuts.
- **SEC Filing Regulatory Retrieval:** Parsing complex 10-K/10-Q filings into fine-grained, notation-preserving financial chunks, indexed via hybrid lexical BM25 and dense neural embeddings fused with Reciprocal Rank Fusion (RRF).
- **Answer-Quality vs. Factual Verification Audits:** Benchmarking model completion quality while explicitly establishing that stylistic quality scoring is not a proxy for regulatory factual truth.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FINVERITAS ARCHITECTURE                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   STAGE 1: DISCOURSE EVASION CLASSIFICATION (EvasionBench)                             │
│   ┌────────────────────┐     ┌─────────────────────────────────────────────────────┐   │
│   │ Analyst Question   │ ──> │ DeBERTa-v3-xsmall (384-token pair)                  │   │
│   │ Executive Answer   │     │  - Direct (52.3%)                                   │   │
│   └────────────────────┘     │  - Intermediate (44.0%)                             │   │
│                              │  - Fully Evasive (3.7%) [F1: 0.5065, Recall: 62.9%] │   │
│                              └──────────────────────────┬──────────────────────────┘   │
│                                                         │ Flagged / Question Extracted │
│                                                         ▼                              │
│   STAGE 2: SEC REGULATORY RETRIEVAL & EVIDENCE LOCALIZATION (FinanceBench)             │
│   ┌────────────────────┐     ┌─────────────────────────────────────────────────────┐   │
│   │ Raw SEC Filings    │ ──> │ Fine-Grained Sliding Window Chunking                │   │
│   │ (10-K / 10-Q PDFs) │     │ (750-char window, 150-char overlap, notation-safe)  │   │
│   └────────────────────┘     └──────────────────────────┬──────────────────────────┘   │
│                                                         │ 37,118 Corpus Chunks         │
│                                                         ▼                              │
│                              ┌─────────────────────────────────────────────────────┐   │
│                              │ Advanced Hybrid Retrieval Engine                    │   │
│                              │  - Lexical BM25 (Okapi k1=1.5, b=0.75)              │   │
│                              │  - Dense Neural (all-MiniLM-L6-v2, 384-d, cached)   │   │
│                              │  - Reciprocal Rank Fusion (RRF, k=60)               │   │
│                              │  - Chunk-to-Page Deduplication                      │   │
│                              └──────────────────────────┬──────────────────────────┘   │
│                                                         │ Evidence Pages / Passages    │
│                                                         ▼                              │
│   STAGE 3: VERIFICATION & CLAIM GROUNDING (Under Active Development)                   │
│   ┌────────────────────┐     ┌─────────────────────────────────────────────────────┐   │
│   │ Grounded Passages  │ ──> │ Answer-Quality Classification (Predicts Label)      │   │
│   │ Executive Claims   │     │  *CAUTION: Does not establish factual truth         │   │
│   └────────────────────┘     │ Proposed: Numerical Symbolic Engine & NLI Fact-Check│   │
│                              └─────────────────────────────────────────────────────┘   │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Experimental Results

### 1. NLP Evasion Classification (EvasionBench Held-Out Test Set, $N=3,346$)

All models evaluate on a held-out test partition with zero question overlap relative to training:

| Model / Configuration | Input Mode | Max Tokens | Accuracy | Macro-F1 | Weighted-F1 | Fully Evasive F1 | Fully Evasive Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | `qa` | — | 52.30% | 0.2289 | 0.3592 | 0.0000 | 0.00% (0/124) |
| **Fair TF-IDF + Logistic Reg.** | `qa` | — | 64.49% | 0.5686 | 0.6482 | 0.4017 | 45.97% (57/124) |
| **DeBERTa-v3-xsmall (256t)** | `qa` | 256 | 63.81% | 0.5941 | 0.6321 | 0.4834 | **64.52% (80/124)** |
| **DeBERTa-v3-xsmall (384t)** | `qa` | 384 | 63.87% | **0.5959** | 0.6328 | **0.5065** | **62.90% (78/124)** |

- **Truncation Mechanics:** Increasing the token limit from 256 to 384 reduced test pair truncation from **45.13% to 20.44%**, boosting Fully-Evasive F1 by $+26.1\%$ relative over the TF-IDF baseline (paired bootstrap $p = 0.004$).

---

### 2. Input Ablations & Shortcut Learning Audits (Checkpoint 06 & 06A)

To determine whether neural classifiers learn true conversational interaction or exploit unilateral answer cues, we audited input ablations and answer-template overlap:

| Model / Input Mode | Evaluated Input | Accuracy | Macro-F1 | Fully Evasive F1 | Fully Evasive Recall | Fully Evasive Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: Question-Only** | Analyst Question Only | 61.81% | 0.4193 | **0.0000** | 0.00% (0/124) | 0.00% |
| **Model B: Answer-Only** | Executive Answer Only | **67.51%** | **0.6251** | **0.5169** | 49.19% (61/124) | **54.46%** |
| **Model C: Question + Answer** | Full Q&A Turn | 63.87% | 0.5959 | 0.5065 | **62.90% (78/124)** | 42.39% |

#### Scientific Findings & Interpretation Governance
1. **Question-Only Minority Behavior:** Question-Only DeBERTa obtained 0 recall and 0 F1 on `fully_evasive` in this evaluated configuration. However, this does not establish that questions contain zero predictive information across all classes or models: Question-Only achieved 64.45% F1 on `direct` and 61.35% F1 on `intermediate` (61.81% accuracy), demonstrating non-trivial predictive signal for non-evasive categories.
2. **Aggregate Macro-F1 vs. Evasive Recall Trade-Off:** Answer-Only achieved higher overall Macro-F1 than Question+Answer (**0.6251 vs. 0.5959**, $+0.0292$, $p = 0.008$, 95% CI `[+0.0044, +0.0510]`). However, Question+Answer achieved substantially higher fully-evasive recall: **62.90% (78/124)** compared with **49.19% (61/124)** for Answer-Only (+13.71% relative recall advantage; 17 additional evasive instances identified). Answer-Only and Question+Answer exhibit an empirical trade-off between aggregate performance and minority evasion sensitivity. Question text cannot be characterized as a universal distractor; its effect is limited to this specific model, training setup, and evaluation cohort.
3. **Answer-Template Leakage Diagnostics:** On the standard question-disjoint split, 32 unique normalized answers cross from training into test (35 test samples, 1.05% of test data) with a **97.14% label agreement rate**. On a **strict normalized-answer-disjoint split** (where zero identical answers cross partitions), the Question+Answer model obtained Fully-Evasive F1 of **0.3977** (recall: **28.23%**, 35/124 caught).
4. **Distinct Evaluation Populations:** The standard question-disjoint and strict answer-disjoint splits constitute mathematically distinct evaluation populations with different sample subsets. While the observed difference is consistent with a shortcut-learning concern, it **does not isolate the causal contribution of memorization** from broader population distribution shifts. Both split benchmarks are preserved.
5. **Pseudo-Label Nature:** All EvasionBench labels (`eva4b_label`) are model-generated pseudo-labels from Eva-4B-V2 (arXiv:2601.09142), not human annotations. Reported scores represent pseudo-label agreement, not human gold truth.

---

### 3. SEC Regulatory Retrieval & Integrity Audits (FinanceBench, $N=150$)

FinanceBench benchmarks question answering over complex corporate SEC filings (10-K, 10-Q):

| Retriever Configuration | Granularity | Document Recall@1 | Document Recall@5 | Evidence Page Recall@1 | Evidence Page Recall@5 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Word TF-IDF Baseline** | Full Page | 42.67% | 53.33% | 4.67% | 10.67% | 0.284 |
| **BM25 Page Baseline** | Full Page | 48.00% | **59.33%** | 6.00% | **12.67%** | **0.327** |
| **BM25 Chunk Baseline** *(Preliminary)* | 750-char Chunk | 48.67% | **60.00%** | 6.67% | **14.00%** | **0.334** |

- **Integrity Audit (Checkpoint 05):** Identified that evaluating only on covered/eligible questions artificially inflated document recall to 75.42%. Enforced unified reporting over the full 150-question benchmark. Identified and resolved front-matter offsets (3.3% of annotations) where PDF index differed from SEC filing report page numbering.

---

### 4. FinanceBench Answer-Quality Classification (Checkpoint 06)

Evaluated 2,400 model completions across 150 unique question groups using strict 5-fold grouped-KFold cross-validation (zero question leakage):

| Configuration | Input Mode | Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | Mode Class | 47.29% | 0.2140 | 0.3048 |
| **Model A: Question-Only** | Question Only | 40.71% | 0.3727 | 0.4035 |
| **Model B: Answer-Only** | Model Answer Only | **78.21%** | **0.7704** | **0.7797** |
| **Model C: Question + Answer** | Question + Answer | 77.42% | 0.7602 | 0.7719 |

> [!WARNING]
> **Methodological Boundary: Answer Quality $\ne$ Factual Verification**
>
> Answer-quality classification predicts dataset quality labels (e.g., whether a completion answered the question or refused), **NOT factual correctness against ground-truth SEC filings**. A fluent, convincing model completion can score as "high quality" while containing hallucinated financial figures. Grounded verification requires cross-referencing against SEC filings via symbolic numerical validation.

---

### 5. Advanced Hybrid Retrieval Architecture (Checkpoint 07)

Implemented an advanced hybrid retrieval engine combining lexical BM25, dense neural embeddings, Reciprocal Rank Fusion (RRF), and chunk-to-page deduplication:
- **Financial Chunking:** 750-character window with 150-character overlap, engineered to preserve decimal places, currency symbols (`$`), percentages (`%`), and negative parenthetical accounting notation (`(120)`). Total corpus: 37,118 chunks across 150 filings.
- **Dense Neural Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional embeddings.
- **Reciprocal Rank Fusion (RRF, $k=60$):** Blends BM25 and dense retrieval score ranks:
  $$RRF\_Score(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + \text{rank}_m(d)}$$
- **Production Resilience:**
  - **Resumable Batch Embedding Cache:** Disk-backed `.npy` batches in `artifacts/advanced_retrieval/cache/` with JSON manifest.
  - **SHA-256 Invalidation & Auto-Regeneration:** Automatically verifies batch integrity and regenerates corrupted `.npy` files.
  - **CPU Safety Defaults:** Strict `device="cpu"` fallback to prevent OOM errors or GPU device contention.
  - **Staged 256-Chunk CPU Smoke Test:** Successfully verified on 256 real SEC filing chunks (16 batches of 16), confirming 384-dimensional geometry, zero cache corruption, instant resume (0 ms latency), and plausible passage retrieval.

---

## Component Implementation Status Matrix

| Component | Checkpoint | Status | Evaluated Metric / Deliverable |
| :--- | :--- | :--- | :--- |
| **EvasionBench Data Audit** | Chkpt 01 | **Implemented & Verified** | 16,726 samples, zero malformed records |
| **TF-IDF Evasion Baseline** | Chkpt 02/05 | **Implemented & Verified** | Macro-F1: 0.5686, Fully Evasive F1: 0.4017 |
| **DeBERTa-v3 Evasion Classifier (256t)** | Chkpt 04 | **Implemented & Verified** | Macro-F1: 0.5941, Fully Evasive F1: 0.4834 |
| **DeBERTa-v3 Evasion Classifier (384t)** | Chkpt 05 | **Implemented & Verified** | Macro-F1: 0.5959, Fully Evasive F1: 0.5065 |
| **Input Ablations & Paired Bootstrap** | Chkpt 06/06A | **Implemented & Verified** | Answer-Only Macro-F1: 0.6251, QA Recall: 62.90% |
| **Answer-Template Leakage Audit** | Chkpt 06/06A | **Implemented & Verified** | 32 overlapping answers, 97.14% label agreement |
| **Strict Answer-Disjoint Split** | Chkpt 06/06A | **Implemented & Verified** | Fully Evasive F1: 0.3977 (Distinct population) |
| **FinanceBench Evidence Retrieval** | Chkpt 04 | **Implemented & Verified** | BM25 Doc Recall@5: 59.33%, Page Recall@5: 12.67% |
| **FinanceBench Integrity Audit** | Chkpt 05 | **Implemented & Verified** | Front-matter offsets corrected; $N=150$ total benchmark |
| **Answer-Quality Classification** | Chkpt 06 | **Implemented & Verified** | Answer-Only Macro-F1: 0.7704 (Not factual verifier) |
| **Financial Chunking Engine** | Chkpt 07 | **Implemented & Verified** | 37,118 chunks, financial notation safe |
| **Chunk-Level BM25 Retrieval** | Chkpt 07 | **Implemented & Verified** | Doc Recall@5: 60.00%, Page Recall@5: 14.00% |
| **Resumable Dense Batch Cache** | Chkpt 07 | **Implemented & Verified** | SHA-256 checks, corruption recovery, CPU safe |
| **Reciprocal Rank Fusion (RRF)** | Chkpt 07 | **Implemented & Verified** | Math & tie-breaking verified across 70 tests |
| **256-Chunk CPU Smoke Test** | Chkpt 07 | **Implemented & Verified** | 16 batches, verified embeddings & cache reuse |
| *Full 37,118 Chunk Dense Encoding* | Chkpt 07 | *Gated / Ready to Run* | Code complete; requires dedicated CPU/GPU run |
| *Cross-Encoder Neural Reranker* | Proposed | *Architecture Designed* | Staged for future deployment |
| *Symbolic Arithmetic Validator* | Proposed | *Architecture Designed* | Staged for future deployment |

---

## Repository Structure

```text
finveritas/
├── README.md                                  # Executive summary, benchmark tables & architecture
├── pyproject.toml                             # Package configuration, dependencies & tool configs
├── .gitignore                                 # Git exclusions with explicit artifact whitelists
│
├── src/finveritas/
│   ├── nlp_evasion/                           # Evasion taxonomy, DeBERTa modeling, ablations & audits
│   │   ├── fair_benchmark.py                  # Group-disjoint splitting & shared benchmark protocol
│   │   ├── shortcut_audit.py                  # Answer-template frequency & strict disjoint split
│   │   └── transformer_classifier.py          # DeBERTa dataset collation, training loop & metrics
│   ├── rag_indexing/                          # SEC filing ingestion, chunking & hybrid retrieval
│   │   ├── chunking.py                        # Sliding-window chunker with financial notation safety
│   │   ├── evaluation.py                      # Retrieval metrics (Recall@K, MRR, NDCG) & bootstrap
│   │   ├── financebench_data.py               # FinanceBench dataset loader & PDF text extractor
│   │   └── retrievers.py                      # BM25, TF-IDF, Dense (all-MiniLM) & Hybrid RRF
│   └── verification/                          # Claim verification & answer quality
│       └── answer_quality.py                  # Answer quality grouped-CV & multiclass metrics
│
├── configs/                                   # Non-secret configuration files
│   ├── datasets/financebench.yaml             # FinanceBench file paths & evaluation settings
│   └── retrieval/advanced_retrieval.yaml      # Hybrid retrieval, RRF k=60, cache directories
│
├── scripts/                                   # Standalone command-line entry points
│   ├── predict_evasion.py                     # CLI for local evasion prediction (TF-IDF & DeBERTa)
│   ├── predict_transformer.py                 # Standalone neural transformer evasion CLI
│   ├── run_fair_benchmark.py                  # Fair baseline & 384-token ablation training
│   ├── run_shortcut_audit.py                  # Input ablation & strict disjoint split training
│   ├── evaluate_financebench_retrieval.py     # Evaluation CLI for page-level retrieval
│   ├── evaluate_advanced_retrieval.py         # Evaluation CLI for chunk-level hybrid retrieval
│   ├── test_cpu_256_chunks.py                 # Staged CPU smoke test (256 real SEC chunks)
│   ├── train_answer_quality_baselines.py      # Answer-quality grouped-CV training CLI
│   └── audit_evidence_alignment.py            # PDF page-index offset and alignment auditor
│
├── docs/research/                             # Detailed research papers & scientific audits
│   ├── finveritas_project_report.md           # Master consolidated research & project report
│   ├── nlp_shortcut_audit.md                  # Shortcut audit report with corrected claims
│   ├── nlp_shortcut_audit_errata.md           # Formal errata & claim qualification document
│   ├── fair_nlp_comparison.md                 # Baseline vs 384-token DeBERTa benchmark
│   ├── transformer_nlp_results.md             # Initial DeBERTa evasion classifier results
│   ├── financebench_retrieval_results.md      # Baseline document & page retrieval report
│   ├── financebench_integrity_audit.md        # Front-matter offset & benchmark audit report
│   ├── financebench_answer_quality_results.md # Answer-quality grouped-CV benchmark report
│   └── advanced_retrieval_recovery.md         # Hybrid RRF architecture & smoke test report
│
├── artifacts/                                 # Version-controlled benchmark metrics & configurations
│   ├── initial_baseline/                      # Checkpoint 02 TF-IDF metrics & config
│   ├── transformer_nlp/                       # Checkpoint 04 DeBERTa metrics & config
│   ├── nlp_comparison/                        # Checkpoint 05 Fair benchmark metrics & config
│   ├── nlp_shortcut_audit/                    # Checkpoint 06 Input ablation & split metrics
│   ├── financebench_retrieval/                # Checkpoint 04/05 Retrieval metrics & audit logs
│   ├── answer_quality/                        # Checkpoint 06 Answer-quality metrics & config
│   └── advanced_retrieval/                    # Checkpoint 07 Hybrid retrieval configs
│
└── tests/unit/                                # Automated verification test suite (70 tests)
    ├── test_baselines.py                      # TF-IDF baseline and leakage-free split tests
    ├── test_dataset_audit.py                  # EvasionBench schema and integrity tests
    ├── test_transformer_classifier.py        # Tokenization, truncation and model forward tests
    ├── test_nlp_comparison.py                # Answer normalization and bootstrap tests
    ├── test_nlp_shortcut_audit.py             # Disjoint split, template audit & report integrity
    ├── test_financebench_retrieval.py         # BM25, TF-IDF and page evaluation tests
    ├── test_answer_quality.py                 # Answer quality schema and grouped-KFold tests
    └── test_advanced_retrieval.py             # Chunking, dense cache, RRF fusion & CPU tests
```

---

## Quickstart & Reproducibility Guide

### 1. Environment Installation
```bash
git clone https://github.com/aayan-garg/NLP.git
cd NLP
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core framework and development dependencies
pip install -e ".[dev,nlp]"
```

### 2. Run Test Suite
Verify the full test suite across both NLP and Retrieval tracks (70 tests, 100% pass rate):
```bash
pytest -v
```

### 3. Local Evasion Prediction CLI (CPU or GPU)
Run predictions locally without network access:
```bash
# Baseline TF-IDF Model (instant CPU):
python scripts/predict_evasion.py \
  --question "What was Q3 organic revenue growth?" \
  --answer "Organic revenue grew 4.2% across North America."

# Neural Transformer Model (DeBERTa-v3):
python scripts/predict_evasion.py --transformer \
  --question "Can you break down operating margins by segment?" \
  --answer "We are very pleased with overall operating performance and see strong momentum."
```

### 4. Staged Hybrid Retrieval Smoke Test
Validate the hybrid retrieval engine on 256 real SEC filing chunks (CPU-safe, completes in ~1.2s):
```bash
python scripts/test_cpu_256_chunks.py
```

### 5. Answer-Quality Classification
Evaluate answer-quality classification under 5-fold grouped-CV:
```bash
python scripts/train_answer_quality_baselines.py
```

---

## Limitations & Governance Principles

1. **Pseudo-Labels vs. Human Consensus:** EvasionBench labels are generated by Eva-4B-V2. Reported scores represent alignment with model-generated pseudo-labels rather than validated human ground truth.
2. **Evaluation Population Integrity:** Standard question-disjoint and strict answer-disjoint splits represent distinct evaluation populations. Metric differences reflect both partition distribution shift and shortcut mitigation; they cannot be treated as a pure causal measurement of template memorization.
3. **Retrieval Denominator Governance:** Benchmarks must be reported over the full 150-question corpus ($N=150$) rather than eligible-only subsets to penalize missing documents transparently.
4. **Answer Quality is Not Truth:** Fluent model answers must be independently cross-referenced against audited SEC filings via symbolic numerical validation before being relied upon by financial analysts.
