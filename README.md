# FinVeritas

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

**FinVeritas** is an end-to-end financial intelligence framework designed to detect managerial question-evasion during corporate earnings-call Q&A sessions and verify extracted factual claims against official SEC regulatory filings (10-K, 10-Q, 8-K).

---

## Repository Structure

The project follows a clean, scalable `src/` layout with strict directory ownership:

```text
finveritas/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── src/
│   └── finveritas/
│       ├── __init__.py
│       ├── nlp_evasion/       # Evasion dataset handling, classification, claim extraction & calibration
│       ├── rag_indexing/      # SEC filing parsing, financial table representation, chunking & retrieval
│       ├── crag_engine/       # Corrective RAG: retrieval evaluation, web search fallback & table routing
│       ├── verification/      # Numerical validation, sandboxed code execution & NLI verification
│       └── api/               # Application entry points, CLI, and service interfaces
│
├── tests/
│   ├── unit/                  # Fast, isolated unit tests
│   ├── integration/           # Multi-component & end-to-end integration tests
│   └── test_smoke.py          # Environment sanity & package layout smoke test
│
├── data/
│   ├── raw/                   # Immutable downloaded datasets (excluded from Git)
│   ├── interim/               # Intermediate transformed representations (excluded from Git)
│   └── processed/             # Cleaned, ready-for-modeling datasets (excluded from Git)
│
├── configs/                   # Non-secret configuration files
├── scripts/                   # Thin command-line utilities only (no business logic)
├── benchmarks/                # Reproducible evaluation entry points & benchmark definitions
├── docs/
│   ├── architecture/          # Target architecture and component specifications
│   ├── decisions/             # Architecture Decision Records (ADRs)
│   └── research/              # Source-grounded research reports & dataset provenance
│
└── artifacts/                 # Generated experiment runs, predictions & reports (excluded from Git)
```

---

## Directory Ownership & Separation of Concerns

- **`src/finveritas/nlp_evasion/`**: Focuses on discourse analysis, question-evasion taxonomy (direct, intermediate, fully evasive), model inference, and claim extraction.
- **`src/finveritas/rag_indexing/`**: Focuses on SEC document parsing, HTML/XBRL table parsing, chunking, and dense/hybrid vector index construction.
- **`src/finveritas/crag_engine/`**: Evaluates retrieval confidence, handles query rewriting, footnote verification, and corrective retrieval fallback.
- **`src/finveritas/verification/`**: Implements symbolic/arithmetic validation of numerical claims, safe Python execution for financial formula checking, and NLI verification models.
- **`src/finveritas/api/`**: Thin presentation and interface layer.
- **`scripts/`**: Strictly contains thin command-line entry points. Reusable business logic must be imported from `src/finveritas/`.
- **`tests/`**: Automated verification test suite.

---

## Quickstart & Verification

### 1. Environment Setup
Install the package in editable mode:
```bash
pip install -e .
```

### 2. Run Test Suite
Verify environment, baseline classification, and audit invariants:
```bash
python -m pytest -q
```

---

## NLP Evasion Baseline & Local Inference CLI

### Empirical Baseline Summary (Held-Out Test Split, N = 3,346)
- **Majority-Class Baseline:** Accuracy: 52.30%, Macro-F1: 0.2289, Weighted-F1: 0.3592
- **TF-IDF + Balanced Logistic Regression:** Accuracy: 64.20%, Macro-F1: 0.5759, Weighted-F1: 0.6430
  - Fully Evasive Class F1: 0.4302 (Recall: 45.97%, Precision: 40.43%)
  - Training Time: ~36.83 seconds on CPU (zero GPU/CUDA utilized)

### Methodological & Scientific Governance Disclaimers
1. **Pseudo-Labels:** All dataset labels (`eva4b_label`) are model-generated pseudo-labels produced by Eva-4B-V2, not human ground truth.
2. **Agreement Metric:** Reported evaluation measures agreement with Eva-4B-V2 pseudo-labels, not independently verified real-world evasion-detection performance.
3. **Leakage Boundaries:** Group-aware splitting guarantees **zero normalized-question overlap between train and test**; however, company-level and temporal leakage have not been ruled out due to omitted metadata.
4. **No SOTA Claim / Gold Benchmark Unreleased:** The published 1,000-sample human gold evaluation set is private and unavailable locally; baseline scores must not be claimed as state-of-the-art or compared directly against human gold evaluation.

### Local Inference CLI
A standalone CLI is provided to train and run question-evasion predictions locally without GPU or network access:

1. **Train and persist model artifact:**
   ```bash
   python scripts/predict_evasion.py --train
   ```
   *Saves fitted model to `artifacts/initial_baseline/model.joblib` (excluded from version control).*

2. **Run inference on a Q&A pair:**
   ```bash
   python scripts/predict_evasion.py --question "What was Q3 organic revenue growth?" --answer "Organic revenue grew 4.2% across North America."
   ```

*Note: Output class probabilities are uncalibrated soft scores and must not be interpreted as Bayesian posterior probabilities or confidence estimates.*

---

## FinanceBench Evidence Retrieval (Checkpoint 04)

### Benchmark Overview (Open-Source Sample, N = 150)
- **Dataset:** `PatronusAI/financebench` (arXiv:2311.11944), CC BY-NC 4.0 license.
- **Corpus:** 12,013 extracted financial pages across 84 enterprise filings (10-K, 10-Q, 8-K).
- **Coverage:** 100.0% (150 / 150 benchmark questions indexed across source PDFs).
- **Task:** Retrieve gold evidence pages using only raw question text (zero answer or justification leakage).

### Empirical Retrieval Baseline Comparison
| Metric | Word TF-IDF (Cosine) | Okapi BM25 |
| :--- | :--- | :--- |
| **Gold Document Recall@1** | 24.67% (37 / 150) | **35.33%** (53 / 150) |
| **Gold Document Recall@5** | 54.67% (82 / 150) | **59.33%** (89 / 150) |
| **Gold Evidence Page Recall@1** | 4.00% (6 / 150) | **8.67%** (13 / 150) |
| **Gold Evidence Page Recall@5** | 8.67% (13 / 150) | **12.67%** (19 / 150) |
| **Gold Evidence Page MRR** | 0.0589 | **0.0986** |
| **Mean Retrieval Latency (CPU)** | 151.02 ms | **26.06 ms (5.8x faster)** |

### Run Retrieval Evaluation CLI
```bash
# Evaluate TF-IDF and BM25 on local corpus:
python scripts/evaluate_financebench_retrieval.py

# Download missing PDFs and evaluate:
python scripts/evaluate_financebench_retrieval.py --download
```

*Disclaimer: Retrieval evaluation measures page-level evidence locating. Retrieving the gold page does not establish answer generation accuracy or verified financial reasoning.*
