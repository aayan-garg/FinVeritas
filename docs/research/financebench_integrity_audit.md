# FinVeritas: FinanceBench Ground-Truth Integrity Audit & Metric Revalidation

- **Date:** 2026-10-18
- **Checkpoint:** Checkpoint 05B (FinanceBench Ground-Truth Integrity Audit)
- **Status:** Complete, Empirically Audited & Verified (CPU-Only)
- **Branch:** `checkpoint/05-financebench-integrity-audit`
- **Starting Commit:** `11a7979`
- **Evaluator:** FinVeritas Research Team

---

## Executive Summary & Core Verdict

Prior to advancing FinVeritas to hybrid retrieval, neural reranking, or claim verification, a rigorous source-level integrity audit of **FinanceBench** (`patronus-ai/financebench`, arXiv:2311.11944) was executed. Specifically, this audit addresses two critical empirical questions:

1. **Evidence-Page Alignment:** Does the published `evidence_page_num` in FinanceBench exhibit a systematic page-offset anomaly (as suggested by certain third-party audits), or does it accurately track the 0-indexed page layout of the underlying SEC PDFs?
2. **Model Output Evaluation Readiness:** What are the schema characteristics, human correctness labels, refusal rates, and consistency of the 16 released benchmark completion files (`results/`, comprising 2,400 records)?

### Key Audit Findings

1. **Zero Systematic Page Offset ($\Delta = 0$ in 100.0% of Evidence Items):**
   - All **189 evidence annotations** across the 150 benchmark questions were audited against the complete extracted corpus of **12,013 PDF pages** spanning all 84 source documents.
   - **189 / 189 annotations (100.0%)** match the ground-truth evidence text at the exact published `evidence_page_num` under zero-based indexing (`reader.pages[page_index]`).
   - The suspected "+1" or variable offset reported by external auditors is an artifact of confusing **0-based programmatic reader page indices** with **1-based human-facing printed page numbers** or cover sheet counts. No mechanical shifting or page adjustments are required or permitted in FinVeritas.
2. **Clean, Human-Annotated Model Output Benchmark:**
   - All **16 model completion JSONL files** (2,400 records) were acquired, validated, and categorized.
   - **Zero duplicates** and **zero missing labels**: All 2,400 records contain verified human-evaluated labels.
   - **Label Distribution:** 
     - **Correct Answer:** 1,135 (47.29%)
     - **Refusal:** 737 (30.71%)
     - **Incorrect Answer:** 528 (22.00%)
   - These 2,400 completions provide an immediate, leak-free testbed for the upcoming FinVeritas claim-verification engine.
3. **Revalidated Lexical Retrieval Baselines:**
   - Under confirmed zero-offset ground truth, Okapi BM25 and Word TF-IDF baselines were re-evaluated across all 12,013 pages.
   - **BM25:** Document Recall@5 = **59.33%**, Page Recall@5 = **12.67%**, Page MRR = **0.0986**, Latency = **26.06 ms**.
   - **TF-IDF:** Document Recall@5 = **54.67%**, Page Recall@5 = **8.67%**, Page MRR = **0.0589**, Latency = **151.02 ms**.

---

## 1. Evidence-Page Alignment Audit

### 1.1. Motivation & The Alleged Offset Issue

In third-party evaluations of financial retrieval benchmarks, discrepancies have frequently been reported where annotated evidence pages did not match the text extracted from SEC filings. Hypotheses included:
- Inconsistent zero-indexing vs. one-indexing across different annotators.
- Cover pages or exhibit slips being omitted during SEC Edgar ingestion.
- Multi-page evidence snippets being indexed only by ending or beginning page.

To resolve this question definitively for FinVeritas, we implemented an automated alignment verification engine (`scripts/audit_evidence_alignment.py`) that matches the gold text snippets against the extracted PDF text.

### 1.2. Verification Protocol

For each of the 189 evidence items across the 150 questions:
1. **Document Resolution:** Resolve the target document name with priority:
   $$\text{doc\_name} = \text{ev.get('doc\_name')} \lor \text{ev.get('evidence\_doc\_name')} \lor \text{question.get('doc\_name')}$$
2. **Candidate Evaluation:**
   - Extract raw and normalized text from `doc_pages[doc_name][annotated_page]`.
   - Compute token-level overlap: $\text{Recall}_{\text{tok}} = \frac{|\text{Tokens}_{\text{evidence}} \cap \text{Tokens}_{\text{page}}|}{|\text{Tokens}_{\text{evidence}}|}$.
   - Test for verbatim normalized substring presence.
3. **Neighborhood & Document Search:**
   - Scan neighborhood pages $\delta \in \{-2, -1, +1, +2\}$.
   - If no match reaches $\ge 70\%$ token recall, scan the entire document.
   - Calculate empirical offset: $\Delta = \text{best\_matching\_page} - \text{annotated\_page}$.

### 1.3. Empirical Audit Results

| Audit Category | Count | Percentage | Verdict |
| :--- | :--- | :--- | :--- |
| **Total Evidence Annotations Audited** | 189 | 100.0% | Complete Benchmark Sample |
| **Exact Substring / Token Match at Annotated Page ($\Delta = 0$)** | **189** | **100.0%** | **Verified 0-Indexed** |
| **Neighborhood Match ($\Delta \in \{-2, -1, +1, +2\}, \Delta \neq 0$)** | 0 | 0.0% | None detected |
| **Distant Page Match ($|\Delta| > 2$)** | 0 | 0.0% | None detected |
| **Missing Document / Unlocatable** | 0 | 0.0% | None detected |

```text
Offset Distribution:
  offset = 0: 189 (100.0%)
  All other offsets: 0
```

### 1.4. Conclusion on Page Alignment

The FinanceBench dataset's `evidence_page_num` field is **100% aligned with 0-indexed PDF page extraction**. Any workflow using standard libraries such as `pypdf`, `pymupdf`, or `pdfplumber` where page indices start at `0` will access the exact ground-truth evidence page without modification.

---

## 2. Audit of Human-Annotated Model Outputs (`results/`)

### 2.1. Dataset Provenance & Scope

The official FinanceBench repository provides a `results/` directory containing execution traces and human-annotated evaluations of 16 frontier language model and RAG configurations responding to the 150 benchmark questions.

All 16 JSONL files were downloaded to `data/raw/financebench/results/` (Git-ignored) and audited via `scripts/audit_model_results.py`.

### 2.2. Schema & Data Consistency

Across all 16 files and 2,400 individual records:
- **Consistent Schema (8 keys):**
  `['financebench_id', 'model_name', 'eval_mode', 'temp', 'question', 'gold_answer', 'model_answer', 'label']`
- **Duplicate Records:** **0** (each file contains exactly 150 unique questions).
- **Missing / Null Labels:** **0** (100% human-annotated label coverage).
- **Average Model Evaluations per Question:** **16.0** (perfectly balanced matrix).

### 2.3. Overall Label Breakdown

| Human Evaluation Label | Record Count | Percentage | Interpretation for Verification |
| :--- | :--- | :--- | :--- |
| **Correct Answer** | 1,135 | 47.29% | Verified factual financial claims grounded in filings. |
| **Refusal** | 737 | 30.71% | Model explicitly abstained ("insufficient information"). |
| **Incorrect Answer** | 528 | 22.00% | Hallucinations, calculation errors, or contradicted claims. |
| **Total Completions** | **2,400** | **100.0%** | Comprehensive Multi-Model Evaluation Set |

### 2.4. Performance by Model & Mode

| Model Configuration | Mode | Total Rows | Correct | Refusal | Incorrect | Accuracy (excl. refusal) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gpt-4-1106-preview-direct` | Direct | 150 | 88 (58.7%) | 38 (25.3%) | 24 (16.0%) | 78.6% |
| `gpt-4-1106-preview-retrieval` | RAG | 150 | 102 (68.0%) | 24 (16.0%) | 24 (16.0%) | 81.0% |
| `claude-2-direct` | Direct | 150 | 74 (49.3%) | 49 (32.7%) | 27 (18.0%) | 73.3% |
| `claude-2-retrieval` | RAG | 150 | 85 (56.7%) | 41 (27.3%) | 24 (16.0%) | 78.0% |
| `gpt-3.5-turbo-direct` | Direct | 150 | 51 (34.0%) | 62 (41.3%) | 37 (24.7%) | 58.0% |
| `llama-2-70b-chat-direct` | Direct | 150 | 42 (28.0%) | 68 (45.3%) | 40 (26.7%) | 51.2% |

*(Full summary of all 16 configurations is recorded in `artifacts/financebench_retrieval/integrity_audit.json`.)*

### 2.5. Significance for FinVeritas Verification Pipeline

In the FinVeritas architecture, natural-language answers must be decomposed into atomic financial claims and verified against retrieved SEC evidence. The 528 `Incorrect Answer` records and 1,135 `Correct Answer` records provide a pre-labeled, gold-standard benchmark to validate our hallucination detection and claim verification classifiers without requiring expensive manual re-annotation.

---

## 3. Revalidated Lexical Retrieval Benchmarks

With evidence page alignment empirically confirmed at $\Delta = 0$, the baseline lexical retrieval metrics are established as verified ground truth:

| Metric | Word TF-IDF (Cosine) | Okapi BM25 | Absolute Difference | Relative Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Gold Document Recall@1** | 24.67% (37 / 150) | **35.33%** (53 / 150) | +10.66% | +43.2% |
| **Gold Document Recall@3** | 43.33% (65 / 150) | **51.33%** (77 / 150) | +8.00% | +18.5% |
| **Gold Document Recall@5** | 54.67% (82 / 150) | **59.33%** (89 / 150) | +4.66% | +8.5% |
| **Gold Evidence Page Recall@1** | 4.00% (6 / 150) | **8.67%** (13 / 150) | +4.67% | +116.8% |
| **Gold Evidence Page Recall@3** | 7.33% (11 / 150) | **10.00%** (15 / 150) | +2.67% | +36.4% |
| **Gold Evidence Page Recall@5** | 8.67% (13 / 150) | **12.67%** (19 / 150) | +4.00% | +46.1% |
| **Gold Evidence Page MRR** | 0.0589 | **0.0986** | +0.0397 | +67.4% |
| **Mean Retrieval Latency** | 151.02 ms | **26.06 ms** | -124.96 ms | **5.8x faster** |

---

## 4. Derived Artifacts & Version Control Integrity

All audit outcomes are committed to the repository in small, structured JSON format:

1. `artifacts/financebench_retrieval/integrity_audit.json`: Complete alignment audit and 16-model completion catalog.
2. `artifacts/financebench_retrieval/metrics_revalidated.json`: Verified benchmark metrics and comparison data.
3. `.gitignore`: Strict exclusion of all raw PDF binaries (`data/raw/financebench/pdfs/*.pdf`, ~158 MB) and raw model completions (`data/raw/financebench/results/*.jsonl`), keeping git history lightweight.

---

## 5. Architectural Recommendations for Subsequent Checkpoints

1. **Retain 0-Indexed Conventions:** Do not apply any page-shift offsets in `src/finveritas/rag_indexing/` or future dense retrieval stages.
2. **Dense & Hybrid Retrieval Necessity:** Lexical BM25's Page Recall@5 of 12.67% demonstrates that lexical matching alone cannot overcome financial vocabulary mismatch (e.g., questions asking about "capital expenditures" when 10-K tables label them "Additions to property, plant and equipment"). Checkpoint 06 should evaluate hybrid lexical-dense retrieval.
3. **Leverage the 2,400 Model Completions:** Use the 528 incorrect and 1,135 correct model answers to benchmark the upcoming NLI / factual verification module (`src/finveritas/verification/`).
