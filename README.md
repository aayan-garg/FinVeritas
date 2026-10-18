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

### 2. Run Smoke Tests
Verify package importability and environment sanity:
```bash
python -m unittest tests/test_smoke.py
```

### 3. Run Acceptance Tests
Verify baseline classification functions:
```bash
python -m unittest tests/unit/test_baseline_acceptance.py
```
