# FinVeritas

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-80%20Total%20%7C%2077%20Passed-brightgreen.svg)]()
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
- **Evidence-Grounded Numerical Claim Verification:** Deterministic, symbolic verification of financial claims against retrieved SEC filing pages using exact `Decimal` arithmetic, unit/scale normalizations, and explicit rounding intervals.

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
│   STAGE 3: VERIFICATION & CLAIM GROUNDING (Evidence-Grounded MVP)                      │
│   ┌────────────────────┐     ┌─────────────────────────────────────────────────────┐   │
│   │ Grounded Passages  │ ──> │ Deterministic Numerical Claim Verifier (Checkpoint  │   │
│   │ Executive Claims   │     │  09 MVP: Decimal math, scale normalization, bounds, │   │
│   └────────────────────┘     │  abstention, official SEC citations)                │   │
│                              │ Answer-Quality Classification (Quality label only)  │   │
│                              │ Proposed: Complex GAAP/Footnote Reconciliation      │   │
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

FinanceBench benchmarks evidence retrieval over corporate SEC filings (10-K, 10-Q). All full-corpus lexical evaluations are reported over the unified 150-question benchmark (12,013 PDF pages, 37,118 chunks):

| Retriever Configuration | Granularity | Candidate Pool | Document Recall@1 | Document Recall@5 | Evidence Page Recall@1 | Evidence Page Recall@5 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Word TF-IDF Baseline** | Full Page | 12,013 pages | 42.67% | 53.33% | 4.67% | 10.67% | 0.284 |
| **BM25 Page Baseline** | Full Page | 12,013 pages | 35.33% | **59.33%** | 8.67% | **12.67%** | **0.1027** |
| **BM25 Chunk Baseline** (256/64) | 256-token Chunk | 37,118 chunks | 29.33% | **60.00%** | 5.33% | **14.00%** | 0.0887 |

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
> Answer-quality classification predicts dataset quality labels (e.g., whether a completion answered the question or refused), **NOT factual correctness against ground-truth SEC filings**. A fluent, convincing model completion can score as "high quality" while containing hallucinated financial figures. Grounded financial verification is **under active research** and requires cross-referencing against audited SEC filings via symbolic numerical validation.

---

### 5. Advanced Hybrid Retrieval & Bounded Smoke Test (Checkpoints 07 & 08)

Implemented an advanced hybrid retrieval engine combining lexical BM25, dense neural embeddings (`BAAI/bge-small-en-v1.5`), Reciprocal Rank Fusion (RRF), and chunk-to-page deduplication:
- **Financial Chunking:** 256-token sliding window with 64-token overlap, engineered to preserve decimal places, currency symbols (`$`), percentages (`%`), and negative parenthetical accounting notation (`(120)`). Total corpus: 37,118 chunks across 84 SEC filings.
- **Dense Neural Embeddings:** `BAAI/bge-small-en-v1.5` generating 384-dimensional normalized vector representations.
- **Reciprocal Rank Fusion (RRF, $k=60$):** Blends BM25 and dense retrieval score ranks:
  $$RRF\_Score(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + \text{rank}_m(d)}$$
- **Production Resilience:**
  - **Resumable Batch Embedding Cache:** Disk-backed `.npy` batches with JSON manifests and SHA-256 integrity metadata.
  - **CPU Safety Enforcement:** Strict `device="cpu"` fallback to prevent CUDA device contention and memory exhaustion.

#### Controlled Bounded Real-Data Smoke Test (Apples-to-Apples 4-Way Comparison)

> [!NOTE]
> **Smoke Test Scope vs. Full Benchmark**
>
> The dense vector and hybrid RRF comparison below was evaluated on a bounded real-data subset of **4 complete 10-K filings** (`AMD_2022_10K`, `AMERICANEXPRESS_2022_10K`, `BOEING_2022_10K`, `AMCOR_2023_10K`), representing **727 pages** and **2,051 chunks** across **25 real financial questions**.
>
> All 4 retrievers searched the **exact same candidate pool and queries**. This bounded evaluation is strictly a **smoke test**, not a full 150-question / 84-document benchmark. Full 37,118-chunk dense embedding is staged for dedicated background execution.

| Retrieval System | Candidates | Document Recall@5 | Evidence Page Recall@5 | Evidence Page MRR | Page nDCG@5 | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **(1) Page BM25** | 727 pages | 96.00% | 20.00% | 0.1217 | 0.1372 | **1.3 ms** |
| **(2) Chunk BM25** (256/64) | 2,051 chunks | 92.00% | 12.00% | 0.1002 | 0.0972 | 2.8 ms |
| **(3) Dense BGE** (`bge-small-en-v1.5`) | 2,051 chunks | **100.00%** | **44.00%** | **0.2704** | **0.3084** | 29.1 ms |
| **(4) Equal-Weight Hybrid RRF** ($k=60$) | 2,051 chunks | **100.00%** | **20.00%** | 0.1780 | 0.1755 | 31.9 ms |

#### Scientific Findings & Statistical Governance
1. **Dense vs. BM25 Semantic Superiority:** Dense retrieval significantly outperformed Chunk BM25 on evidence-page recall (+32.00%, **44.00% vs. 12.00%**, 95% CI `[+16.00%, +52.00%]`, paired bootstrap **$p < 0.001$**). Dense semantic representations overcome vocabulary mismatch in non-verbatim financial tables (e.g., matching "capital expenditures" to "property and equipment additions").
2. **Dense Outperformed Equal-Weight RRF:** Dense retrieval alone outperformed equal-weight Hybrid RRF on this subset (**44.00% vs. 20.00%**). Equal-weight RRF ($k=60$) allowed lower-precision lexical BM25 noise to displace true dense hits in top-5 ranks.
3. **No Claim of Hybrid Superiority:** On this tabular financial subset, dense retrieval proved superior to equal-weight RRF. Hybrid RRF is not claimed to be superior to dense retrieval; weighted RRF ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$) is indicated for future exploration.

---

### 6. Evidence-Grounded Financial Claim Verification MVP (Checkpoint 09)

Checkpoint 09 introduces the first regulatory-grounded financial claim-verification vertical slice in FinVeritas, implementing a deterministic, zero-hallucination verification engine (`src/finveritas/verification/numeric_verifier.py`) and retrieval-grounded CLI (`scripts/verify_claim.py`):

- **Pure Decimal Arithmetic:** Operates entirely over `decimal.Decimal` to eliminate binary floating-point rounding errors.
- **Unit & Scale Canonicalization:** Automatically reconciles scale representations across explicit units (e.g., USD 23.6 billion $\equiv$ USD 23,600 million $\equiv$ USD 23,600,000,000).
- **Dual Rounding Interval & Tolerance:** Computes explicit precision bounds based on stated significant decimal digits ($\Delta_{\text{bounds}} = \frac{1}{2} \cdot 10^{-d} \cdot S_{\text{claim}}$) alongside a 0.5% relative tolerance ($\Delta_{\text{rel}} = |V_{\text{claim}}| \cdot 0.005$).
- **Closed-World Verdict State Machine:**
  - `SUPPORTED`: Sufficiently specific evidence from the relevant filing matches the metric, period, and value within documented tolerance.
  - `CONTRADICTED`: Directly comparable filing evidence for the same metric and period reports a conflicting value outside tolerance.
  - `INSUFFICIENT_EVIDENCE`: Missing evidence, unmatched reporting periods, unit mismatches, conflicting candidates, or parsing ambiguity.
  - *Epistemic Safety Rule*: Never infer truth from retrieval score alone. Never infer contradiction from passage omission.
- **Un-Truncated Page Retrieval:** Connects BM25 retrieval over full SEC filing pages (preserving all financial statement row context and column headers) rather than truncated snippets.
- **Audit-Grade Citations:** Every verdict produces an `EvidenceCitation` detailing document ID, 0-based PDF page index, 1-based filing display page, official SEC filing URL, and matched text snippet.

#### Real SEC Filing Demonstration (AMD 2022 Form 10-K)
Validated on the official AMD 2022 10-K filing from the local FinanceBench corpus:
- **Claim:** AMD FY2022 Net Sales were `$23.6 billion`.
- **Filing Evidence:** Consolidated Statements of Operations (PDF page index 42, display page 43) reports Net Sales of `$23,601 million`.
- **Verdict:** `SUPPORTED` (diff: $1,000,000 USD within $50,000,000 USD precision bounds and $118,000,000 USD relative tolerance).
- **Official SEC Citation:** [AMD 2022 10-K (Page 43)](https://ir.amd.com/sec-filings/content/0000002488-23-000047/0000002488-23-000047.pdf).

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
| **FinanceBench Evidence Retrieval** | Chkpt 04 | **Implemented & Verified** | Full-corpus BM25 Doc Recall@5: 59.33%, Page Recall@5: 12.67% |
| **FinanceBench Integrity Audit** | Chkpt 05 | **Implemented & Verified** | Front-matter offsets corrected; $N=150$ total benchmark |
| **Answer-Quality Classification** | Chkpt 06 | **Implemented & Verified** | Answer-Only Macro-F1: 0.7704 (Quality label only, not factual truth) |
| **Financial Chunking Engine** | Chkpt 07 | **Implemented & Verified** | 37,118 chunks, financial notation safe |
| **Chunk-Level BM25 Retrieval** | Chkpt 07 | **Implemented & Verified** | Full-corpus Doc Recall@5: 60.00%, Page Recall@5: 14.00% |
| **Resumable Dense Batch Cache** | Chkpt 07 | **Implemented & Verified** | SHA-256 checks, corruption recovery, CPU safe |
| **Reciprocal Rank Fusion (RRF)** | Chkpt 07/08 | **Implemented & Verified** | Math, tie-breaking, and empty/missing dense fallback verified |
| **Fast Hybrid RAG Smoke Test** | Chkpt 08 | **Implemented & Verified** | Bounded 4-filing smoke test: Dense Page R@5 44%, RRF 20% ($p < 0.001$) |
| **Deterministic Numerical Claim Verifier** | Chkpt 09 | **Implemented & Verified** | Pure Decimal math, scale normalization, bounds interval, citations |
| **Filing-Grounded Claim Verification CLI** | Chkpt 09 | **Implemented & Verified** | BM25 retrieval over full filing page text, SEC URL citations |
| *Full 37,118 Chunk Dense Encoding* | Staged | *Ready to Run* | Code complete; requires dedicated background CPU job |
| *Cross-Encoder Neural Reranker* | Proposed | *Architecture Designed* | Staged for future deployment |
| *General-Purpose Complex GAAP Verifier* | Proposed | *Under Active Research* | Non-GAAP reconciliations, multi-step math, scanned footnotes |

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
│       ├── answer_quality.py                  # Answer quality grouped-CV & multiclass metrics
│       └── numeric_verifier.py                # Deterministic Decimal claim verifier & citation models
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
│   ├── audit_evidence_alignment.py            # PDF page-index offset and alignment auditor
│   └── verify_claim.py                        # CLI for evidence-grounded financial claim verification
│
├── docs/research/                             # Detailed research papers & scientific audits
│   ├── finveritas_project_report.md           # Master consolidated research & project report
│   ├── numeric_claim_verification_mvp.md      # Grounded claim verification MVP & real filing audit
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
└── tests/unit/                                # Automated verification test suite (80 tests)
    ├── test_baselines.py                      # TF-IDF baseline and leakage-free split tests
    ├── test_dataset_audit.py                  # EvasionBench schema and integrity tests
    ├── test_transformer_classifier.py        # Tokenization, truncation and model forward tests
    ├── test_nlp_comparison.py                # Answer normalization and bootstrap tests
    ├── test_nlp_shortcut_audit.py             # Disjoint split, template audit & report integrity
    ├── test_financebench_retrieval.py         # BM25, TF-IDF and page evaluation tests
    ├── test_answer_quality.py                 # Answer quality schema and grouped-KFold tests
    ├── test_advanced_retrieval.py             # Chunking, dense cache, RRF fusion & CPU tests
    └── test_numeric_verifier.py               # Deterministic verifier, citations, and real filing tests
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
Verify the full test suite across NLP, Retrieval, and Verification tracks (80 tests: 77 passed, 3 skipped):
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

### 6. Evidence-Grounded Financial Claim Verification CLI
Verify financial claims against local SEC filings with BM25 page retrieval and deterministic Decimal math:
```bash
python scripts/verify_claim.py \
  --claim "AMD net sales were $23.6 billion for fiscal 2022" \
  --ticker AMD \
  --period 2022 \
  --metric revenue \
  --value 23.6 \
  --scale billion
```

---

## Limitations & Governance Principles

1. **Pseudo-Labels vs. Human Consensus:** EvasionBench labels are generated by Eva-4B-V2. Reported scores represent alignment with model-generated pseudo-labels rather than validated human ground truth.
2. **Evaluation Population Integrity:** Standard question-disjoint and strict answer-disjoint splits represent distinct evaluation populations. Metric differences reflect both partition distribution shift and shortcut mitigation; they cannot be treated as a pure causal measurement of template memorization.
3. **Retrieval Denominator Governance:** Benchmarks must be reported over the full 150-question corpus ($N=150$) rather than eligible-only subsets to penalize missing documents transparently.
4. **Answer Quality is Not Truth:** Fluent model answers must be independently cross-referenced against audited SEC filings via symbolic numerical validation before being relied upon by financial analysts.
5. **Epistemic Abstention in Claim Verification:** The numerical claim verifier enforces a fail-closed policy. Retrieval omissions, ambiguous tables, or unparseable context trigger `INSUFFICIENT_EVIDENCE`. Contradictions are declared only when directly comparable filing evidence affirmatively conflicts on the exact metric and period.

