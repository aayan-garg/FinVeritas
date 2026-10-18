# FinVeritas Architecture Specification: Repository Layout & Module Boundaries

## Status
- **Date:** 2026-10-18
- **Phase:** Checkpoint 00 (Bootstrap)
- **Status:** Approved

---

## 1. Architectural Vision

FinVeritas is a dual-stage financial intelligence system:
1. **Discourse Stage (`nlp_evasion`):** Evaluates earnings-call Q&A exchanges to detect whether management directly answered an analyst's question, hedged/partially addressed it (intermediate), or fully evaded it.
2. **Verification Stage (`rag_indexing`, `crag_engine`, `verification`):** For non-evasive and partial statements containing factual financial claims, retrieves official SEC regulatory disclosures (10-K, 10-Q, 8-K) and performs numerical and Natural Language Inference (NLI) claim verification.

To prevent architectural debt, all codebase components adhere to a strict `src/` layout with clean boundaries.

---

## 2. Directory Ownership & Component Mapping

| Subsystem / Path | Architectural Responsibility | Allowed Dependencies | Prohibited Content |
| :--- | :--- | :--- | :--- |
| **`src/finveritas/nlp_evasion/`** | Dataset ingestion, question-evasion classification, hedging calibration, claim extraction. | PyTorch, Hugging Face, Scikit-Learn | SEC filing parsers, RAG vector stores, direct API routes |
| **`src/finveritas/rag_indexing/`** | SEC EDGAR document parsing, HTML/XBRL table parsing, chunking, dense/hybrid vector index construction. | LangChain/LlamaIndex, FAISS/Chroma, BeautifulSoup | Evasion model weights, NLI verification logic |
| **`src/finveritas/crag_engine/`** | Corrective RAG: evaluates retrieval sufficiency, triggers query rewrite, handles web search fallback and footnote table routing. | `rag_indexing`, search APIs | Direct GUI code, raw dataset scripts |
| **`src/finveritas/verification/`** | Numerical calculation validation, sandboxed Python code execution for financial formulas, NLI entailment scoring. | SymPy, PyTorch/NLI models | Ingestion logic, presentation endpoints |
| **`src/finveritas/api/`** | CLI entry points, FastAPI/REST services, input validation schemas. | `finveritas.*` subsystems | Heavy ML training loops, internal database manipulations |
| **`scripts/`** | Thin command-line runners and utilities. | `src.finveritas` | Standalone, duplicated business logic |
| **`experiments/`** | Isolated research experiments and legacy spikes (`experiments/legacy/`). | Standalone research scripts | Direct import by production code, inclusion in default test suites |
| **`tests/`** | Automated unit and integration tests. | Standard library unittest, pytest | Non-test production code, legacy spike validation |
| **`docs/`** | Architectural specifications, ADRs, and research documentation. | Markdown | Executable Python binaries |
| **`data/`** | `raw/`, `interim/`, `processed/` data tiers. | Excluded from Git via `.gitignore` | Checked-in production datasets |
| **`artifacts/`** | Generated experiment runs, predictions, confusion matrices. | Excluded from Git via `.gitignore` | Checked-in large binary artifacts |

---

## 3. Package Layout & Import Convention

All internal imports must use the absolute package namespace:
```python
# Correct:
from finveritas.nlp_evasion import compute_class_weights
from finveritas.rag_indexing import SECDocumentParser

# Prohibited:
import nlp_evasion
from ..nlp_evasion import compute_class_weights
```

---

## 4. Checkpoint Implementation Policy

- Subsystems must not be implemented prematurely. In Checkpoint 00, modules remain pure package skeletons with verified `__init__.py` files until their respective development checkpoint is authorized.
- Code from exploratory spikes must be reviewed and inventoried before migration.
