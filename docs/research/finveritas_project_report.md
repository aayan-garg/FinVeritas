# FinVeritas: Comprehensive Project & Research Report

- **Document Version:** 1.0.0 (Consolidated Handoff)
- **Date:** 2026-10-19
- **Project:** FinVeritas — Financial Question-Evasion Detection & SEC Regulatory Verification
- **Author:** FinVeritas Engineering & Research Team
- **Status:** Complete, Consolidated, and Verified

---

## Executive Summary & System Overview

**FinVeritas** is a financial intelligence and verification framework designed to address a critical asymmetry in modern capital markets: **managerial evasiveness during corporate earnings-call Q&A sessions** and the subsequent need to **audit executive claims against regulatory truth in SEC filings (10-K, 10-Q, 8-K)**.

Corporate executives frequently utilize rhetorical techniques—formulaic non-answers, selective omission, semantic hedging, and diversionary optimism—to deflect pointed inquiries from equity research analysts. Downstream financial analysts and automated algorithmic trading systems risk absorbing ungrounded managerial rhetoric as fact unless claims are systematically audited.

FinVeritas tackles this dual challenge through a multi-stage, scientifically grounded architecture:

```
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

## 1. Track 1: Earnings-Call Evasion NLP Pipeline

### 1.1 Problem Formulation & Dataset Provenance
Evasion classification operates on conversational turns from corporate earnings calls. Each instance consists of:
- **Analyst Question ($Q$):** Inquiring about performance drivers, margins, guidance, or operational challenges.
- **Executive Answer ($A$):** Response from corporate executives (CEO, CFO, IR).
- **Target Label ($Y \in \{ \text{direct}, \text{intermediate}, \text{fully\_evasive} \}$):** Evasion classification taxonomy.

**Methodological Governance Disclaimer:**
The primary training and evaluation dataset is **EvasionBench** (16,726 examples). All target labels (`eva4b_label`) are **model-generated pseudo-labels produced by Eva-4B-V2** (a 4-billion parameter Qwen2.5-based fine-tune; arXiv:2601.09142), **not human consensus annotations**. High classification performance reflects fidelity to Eva-4B-V2's scoring behavior rather than validated human ground truth.

### 1.2 Split Integrity & Partitioning Protocol
Corporate earnings calls present severe risks of data leakage: identical analyst questions or recurring firm-specific idioms appear across quarters.
- **Protocol:** Deterministic group-disjoint partitioning based on lowercased, whitespace-normalized question text (`normalized_question`) using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Integrity Guarantee:** Zero question overlap across splits:
  - Training Set: 10,704 samples (64.0%)
  - Validation Set: 2,676 samples (16.0%)
  - Held-Out Test Set: 3,346 samples (20.0%)
  - Class Distribution: Direct ~52.3%, Intermediate ~44.0%, Fully Evasive ~3.7% (124 evasive test samples).

### 1.3 Baseline Benchmarks vs. Neural Modeling

| Model / Configuration | Split / Tokens | Accuracy | Macro-F1 | Weighted-F1 | Direct F1 | Interm. F1 | Fully Evasive F1 | Fully Evasive Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | Held-Out / — | 52.30% | 0.2289 | 0.3592 | 0.6868 | 0.0000 | 0.0000 | 0.00% (0/124) |
| **Fair TF-IDF + Logistic Reg.** | Comparable / — | 64.49% | 0.5686 | 0.6482 | 0.6477 | 0.6565 | 0.4017 | 45.97% (57/124) |
| **DeBERTa-v3-xsmall** | Held-Out / 256 | 63.81% | 0.5941 | 0.6321 | 0.6052 | 0.6738 | 0.4834 | 64.52% (80/124) |
| **DeBERTa-v3-xsmall** | Held-Out / 384 | 63.87% | **0.5959** | 0.6328 | 0.6069 | 0.6742 | **0.5065** | **62.90% (78/124)** |

### 1.4 Truncation Mechanics Audit
At 256 tokens, **45.13% of test sequence pairs were truncated**, disproportionately truncating executive answers where deflections often appear late in the turn. Expanding the maximum context window to 384 tokens reduced truncation to **20.44%**, boosting Fully-Evasive F1 from 0.4834 to 0.5065 (+26.1% relative gain over TF-IDF; bootstrap significance $p = 0.004$).

---

## 2. Input Ablation, Shortcut Learning & Interpretation Governance

### 2.1 Input Ablation Experiments (Held-Out Test Set, $N=3,346$)

To test whether the model genuinely evaluates the question-answer discourse or exploits unilateral linguistic shortcuts, three input configurations were evaluated under identical hyperparameters (384 tokens, seed 42):

| Input Mode | Evaluated Input | Accuracy | Macro-F1 | Fully Evasive F1 | Fully Evasive Recall | Fully Evasive Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A: Question-Only** | Analyst Question Only | 61.81% | 0.4193 | **0.0000** | 0.00% (0/124) | 0.00% |
| **Model B: Answer-Only** | Executive Answer Only | **67.51%** | **0.6251** | **0.5169** | 49.19% (61/124) | **54.46%** |
| **Model C: Question + Answer** | Full Dialogue Turn | 63.87% | 0.5959 | 0.5065 | **62.90% (78/124)** | 42.39% |

### 2.2 Corrected Scientific Interpretations (Checkpoint 06A Errata)

In accordance with [`docs/research/nlp_shortcut_audit_errata.md`](file:///docs/research/nlp_shortcut_audit_errata.md), three essential scientific corrections govern these results:

1. **Question-Only Minority Behavior Bounded:**
   - In the evaluated configuration, Question-Only DeBERTa obtained 0 recall and 0 F1 on `fully_evasive`.
   - **Crucially, this does not establish that questions contain zero predictive information across all classes or models.** Question-Only achieved 64.45% F1 on `direct` and 61.35% F1 on `intermediate` (61.81% accuracy), demonstrating non-trivial predictive signal for non-evasive categories.
2. **Aggregate Macro-F1 vs. Evasive Recall Trade-Off:**
   - Answer-Only achieved higher overall Macro-F1 than Question+Answer (**0.6251 vs. 0.5959**, $+0.0292$, paired bootstrap $p = 0.008$, 95% CI `[+0.0044, +0.0510]`).
   - However, Question+Answer achieved substantially higher fully-evasive recall: **62.90% (78/124)** compared with **49.19% (61/124)** for Answer-Only (+13.71% relative recall advantage; 17 additional evasive instances identified).
   - Answer-Only and Question+Answer therefore exhibit an empirical trade-off: Answer-Only provides higher class-averaged F1 and cleaner direct predictions, whereas Question+Answer is significantly more sensitive to minority evasion events.
   - Question text **must not be described as a universal distractor**; its observed interaction is limited to this specific model, training setup, and evaluation cohort.
3. **Answer-Template Leakage & Split Population Differences:**
   - On the standard question-disjoint split, 32 unique normalized answers cross from training into test (affecting 35 test samples, 1.05% of test data) with a **97.14% label agreement rate** (e.g., *"That's correct"*, *"We don't provide that breakdown"*).
   - On a **strict normalized-answer-disjoint split** (where zero identical answers cross partitions), the Question+Answer model obtained a Fully-Evasive F1 of **0.3977** (recall: **28.23%**, 35/124 caught).
   - **Methodological Boundary:** The standard question-disjoint split ($N=3,346$) and strict answer-disjoint split ($N=3,346$) constitute **mathematically distinct evaluation populations** with different sample compositions. While the degradation is consistent with a shortcut-learning concern, it **does not isolate the causal contribution of memorization** from broader population distribution shifts. Claims that ~10.9 F1 points were directly caused by memorization are unproven and qualified.

---

## 3. Track 2: SEC Regulatory Retrieval (FinanceBench)

### 3.1 Dataset Structure & Ground-Truth Alignment Audit
FinanceBench provides a challenging evaluation ground for financial question answering over SEC filings:
- **Scope:** 150 unique financial questions across corporate SEC filings (10-K, 10-Q).
- **Format:** Complex multi-page financial tables, balance sheets, footnotes, and MD&A disclosures.

**Integrity Audit Findings (Checkpoint 05):**
1. **Front-Matter Offsets:** Discovered that 3.3% of annotations exhibited page-index offsets due to PDF cover pages versus financial statement report page numbering.
2. **Denominator Governance:** Prior evaluations calculated recall only over "eligible/covered" questions, artificially inflating document recall to 75.42%. FinVeritas enforces **total benchmark reporting** ($N=150$ denominator) to penalize missing documents transparently.

### 3.2 Retrieval Baselines (FinanceBench Total Benchmark, $N=150$)

| Retriever | Granularity | Document Recall@1 | Document Recall@5 | Evidence Page Recall@1 | Evidence Page Recall@5 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF Baseline** | Full Page | 42.67% | 53.33% | 4.67% | 10.67% | 0.284 |
| **BM25 Baseline** | Full Page | 48.00% | **59.33%** | 6.00% | **12.67%** | **0.327** |
| **BM25 Chunk Baseline** *(Preliminary)* | 750-char Chunk | 48.67% | **60.00%** | 6.67% | **14.00%** | **0.334** |

---

## 4. Track 3: FinanceBench Answer-Quality Classification

### 4.1 Grouped-CV Benchmark (2,400 Model Completions)
To evaluate model-generated financial responses, 2,400 model completions across 150 unique question groups were evaluated using a strict 5-fold grouped-KFold setup (zero question leakage across folds):

| Configuration | Input Mode | Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | Mode Class | 47.29% | 0.2140 | 0.3048 |
| **Model A: Question-Only** | Question Only | 40.71% | 0.3727 | 0.4035 |
| **Model B: Answer-Only** | Model Answer Only | **78.21%** | **0.7704** | **0.7797** |
| **Model C: Question + Answer** | Question + Answer | 77.42% | 0.7602 | 0.7719 |

### 4.2 ⚠️ Critical Methodological Distinction: Answer-Quality $\ne$ Factual Verification

> [!WARNING]
> **Answer-quality classification predicts dataset quality labels, NOT factual correctness against ground-truth SEC filings.**
>
> In FinanceBench, completions are labeled for perceived stylistic and structural competence (e.g., whether the model attempted the question or refused). An executive answer or model completion can be fluent, coherent, and highly confident—scoring as "high quality"—while containing entirely fabricated revenue figures or hallucinated margin percentages.
>
> True financial verification requires cross-document numerical anchoring and symbolic arithmetic validation against audited SEC filings, not superficial linguistic quality scoring.

---

## 5. Track 4: Advanced Hybrid Retrieval Architecture

### 5.1 System Architecture
Checkpoint 07 implemented an advanced hybrid retrieval engine combining lexical precision with dense semantic representations:

```
Query: "What was Q3 operating cash flow for Microsoft?"
   │
   ├──> Tokenize ─────────> BM25 Scoring (Okapi k1=1.5, b=0.75) ────────┐
   │                                                                     │
   └──> Embed (all-MiniLM) ─> Dense Dot-Product / Cosine Similarity ────┤
                                                                         │
                                                                         ▼
                                                     Reciprocal Rank Fusion (RRF, k=60)
                                                                         │
                                                                         ▼
                                                     Chunk-to-Page Deduplication
                                                                         │
                                                                         ▼
                                                     Top-K Evidence Pages & Passages
```

1. **Sliding-Window Chunking:** 750-character window with 150-character overlap. Specifically engineered to preserve financial notation (currency symbols `$`, percentages `%`, negative parenthetical accounting notation `(120)`). Total corpus: 37,118 chunks across 150 SEC documents.
2. **Dense Neural Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional embeddings.
3. **Reciprocal Rank Fusion (RRF):** Blends ranked lists with parameter $k=60$:
   $$RRF\_Score(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + \text{rank}_m(d)}$$
4. **Chunk-to-Page Deduplication:** Automatically collapses multiple passage hits from the same document page, returning the page at its highest-scoring rank.

### 5.2 Production Engineering & Resilience (Checkpoint 07)
- **Resumable Batch Cache:** Disk-backed `.npy` batch storage in `artifacts/advanced_retrieval/cache/` with JSON manifest indexing chunk hash signatures and model parameters.
- **SHA-256 Invalidation:** Automatically detects if corpus chunk text or configurations change, invalidating only modified batches.
- **Corrupted Batch Recovery:** Identifies corrupted or truncated `.npy` files and automatically re-encodes only the damaged batches without restarting.
- **CPU Safety Defaults:** Defaults strictly to `device="cpu"` to prevent OOM errors or CUDA device contention on constrained hardware.

### 5.3 Staged 256-Chunk CPU Smoke Test
Before initiating full-corpus encoding, the architecture was validated on 256 real SEC filing chunks (16 batches of 16 chunks):
- Batch encoding completed in 1.2 seconds on CPU.
- Verified exact 384-dimensional embedding geometry.
- Verified cache persistence and instant resume on subsequent calls (0 ms latency).
- Confirmed plausible semantic passage retrieval on sample financial queries.

---

## 6. Implementation Status Matrix

| Component | Checkpoint | Status | Evaluated Metric / Result |
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
| *Full 37,118 Chunk Dense Encoding* | Chkpt 07 | *Gated / Ready to Run* | Code complete; requires dedicated CPU/GPU job |
| *Cross-Encoder Neural Reranker* | Proposed | *Architecture Designed* | Staged for future deployment |
| *Symbolic Arithmetic Validator* | Proposed | *Architecture Designed* | Staged for future deployment |

---

## 7. Setup & Reproducibility Guide

### 7.1 Environment Setup
```bash
# Clone and create virtual environment
git clone https://github.com/aayan-garg/NLP.git
cd NLP
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package and dependencies
pip install -e ".[dev,nlp]"
```

### 7.2 Running Tests
```bash
# Run all unit tests (67 unit tests, 3 dataset-skipped tests)
pytest -v
```

### 7.3 Local Inference CLI
```bash
# 1. Run baseline TF-IDF evasion prediction (CPU, instant)
python scripts/predict_evasion.py \
  --question "What was Q3 organic revenue growth?" \
  --answer "Organic revenue grew 4.2% across North America."

# 2. Run neural transformer evasion prediction (DeBERTa-v3)
python scripts/predict_evasion.py --transformer \
  --question "Can you break down operating margins by segment?" \
  --answer "We are very pleased with overall operating performance and see strong momentum."
```

### 7.4 Staged Hybrid Retrieval Smoke Test
```bash
# Execute safe CPU smoke test on 256 real financial filing chunks
python scripts/test_cpu_256_chunks.py
```

### 7.5 Answer-Quality Benchmark
```bash
# Run 5-fold grouped-CV answer quality baselines
python scripts/train_answer_quality_baselines.py
```

---

## 8. Limitations & Governance Summary

1. **Pseudo-Label Dependency:** All evasion scores evaluate agreement with Eva-4B-V2. Before deploying in real financial advisory systems, a statistically powered sample of predictions should undergo human expert validation.
2. **Distinct Evaluation Cohorts:** Cross-split metrics (e.g., question-disjoint vs. answer-disjoint) cannot be directly subtracted as a pure causal measurement of memorization because the sample distributions differ.
3. **Retrieval Granularity:** Finding financial disclosures often requires cross-referencing footnotes with table headers. Passage-level chunking must be supplemented with table-aware structure parsing.
4. **Answer Quality is Not Truth:** Fluent model answers must be independently cross-referenced against filing numbers via symbolic arithmetic validation before being relied upon by financial analysts.
