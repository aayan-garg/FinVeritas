# FinVeritas: FinanceBench Evidence Retrieval Benchmark Results

- **Date:** 2026-10-18
- **Checkpoint:** Checkpoint 04 (FinanceBench Evidence Retrieval)
- **Status:** Complete, Measured & Verified (CPU-Only)
- **Branch:** `checkpoint/04-financebench-retrieval`
- **Author:** FinVeritas Engineering Team

---

## Executive Summary & Core Verdict

This document presents the first empirical, reproducible evaluation of financial evidence retrieval in FinVeritas. Using the publicly released 150-question open-source sample of **FinanceBench** (`patronus-ai/financebench`, arXiv:2311.11944), two CPU-only lexical retrieval baselines were implemented, indexed across **12,013 extracted financial filing pages**, and evaluated under zero target leakage.

### Key Empirical Findings
1. **100% Benchmark Coverage:** All 84 required enterprise SEC filings (10-K, 10-Q, 8-K) were successfully acquired and parsed locally using `pypdf` with AES decryption support (`cryptography`), achieving **150/150 (100.0%) question coverage**.
2. **Lexical Baseline Comparison:** Okapi BM25 decisively outperforms Word TF-IDF across all ranking metrics:
   - **Evidence Page Recall@5:** **12.67%** (BM25) vs. **8.67%** (TF-IDF) — a **+46.1% relative improvement**.
   - **Evidence Page MRR:** **0.0986** (BM25) vs. **0.0589** (TF-IDF) — a **+67.4% relative improvement**.
   - **Document Recall@5:** **59.33%** (BM25) vs. **54.67%** (TF-IDF).
   - **Retrieval Latency:** BM25 retrieves pages in **26.06 ms/query** on CPU, operating **5.8x faster** than sparse matrix TF-IDF (151.02 ms/query).
3. **The Document-to-Page Retrieval Drop-off:**
   While BM25 identifies the correct source document within the top 5 candidates **59.33% of the time**, it pinpoints the exact gold evidence page within top 5 only **12.67% of the time**. In financial 10-Ks spanning 100–300 pages, standard lexical matching heavily favors boilerplate sections (MD&A liquidity discussions, derivative risk disclosures) that repeat query terms across multiple quarters, missing dense numerical balance sheets and cash flow tables.

### ⚠️ Methodological & Governance Disclaimers
1. **Retrieval-Only Evaluation:** This benchmark evaluates document- and page-level retrieval solely. Retrieving the gold evidence page is a necessary prerequisite for financial question answering, but **does not constitute proof of correct numerical calculation, table reasoning, or hallucination prevention**.
2. **Zero Target Leakage:** Neither gold answers, justifications, nor annotated evidence strings were supplied to the retrieval engine or indexed in the corpus. Queries consist strictly of the raw user-facing `question` text.
3. **Verified Page Numbering:** Confirmed that FinanceBench `evidence_page_num` uses a **0-based page index** aligning directly with `reader.pages[page_index]`.

---

## 1. Dataset Provenance & Page Corpus Construction

### 1.1. Benchmark Specifications
- **Source Dataset:** FinanceBench Open-Source Sample (`PatronusAI/financebench`)
- **Repository:** `https://github.com/patronus-ai/financebench`
- **License:** **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International)
- **Benchmark Size:** 150 questions across 84 unique enterprise financial documents (10-K annual reports, 10-Q quarterly reports, 8-K current reports).
- **Target Companies:** 3M, Adobe, Amazon, Activision Blizzard, Amcor, AES, AMD, Apple, Microsoft, Coca-Cola, Pfizer, Walmart, and peer enterprises.

### 1.2. Corpus Extraction & Normalization Statistics
All 84 required source PDFs (totaling 157.9 MB) were acquired from the official repository and parsed using `pypdf`:

| Metric | Measured Value |
| :--- | :--- |
| **Total Source PDFs Targeted** | 84 |
| **Source PDFs Successfully Parsed** | **84 (100.0%)** |
| **PDF Extraction Failures** | **0** |
| **Total Extracted Pages** | **12,013 pages** |
| **Content Pages ($\ge 10$ chars)** | 11,939 pages (99.38%) |
| **Empty / Blank Pages** | 74 pages (0.62%) |
| **Total Benchmark Questions** | 150 |
| **Covered Questions (Source Document Indexed)** | **150 (100.0%)** |
| **Uncovered Questions** | **0 (0.0%)** |

### 1.3. Conservative Text Normalization
To prevent numerical corruption during PDF extraction, `normalize_financial_text` applies:
- NFKC unicode normalization.
- Space standardization (converting non-breaking spaces `\u00a0` and exotic whitespace to ASCII spaces).
- Strict preservation of financial currency symbols (`$`, `€`, `£`, `¥`), decimal points, negative figures in financial parentheses `$(1,749)`, percentages (`%`), scale suffixes (`B`, `M`, `K`), and tabular newline rhythm.

---

## 2. Lexical Retrieval Baselines

Both models were fitted on the complete 12,013-page corpus and evaluated across all 150 questions:

### 2.1. Baseline 1: Word TF-IDF + Cosine Similarity
- **Vocabulary:** Word n-grams `(1, 2)` with financial token pattern preserving symbols and tickers.
- **Filtering:** `min_df=2`, `max_df=0.95`, sublinear term frequency scaling enabled.
- **Scoring:** Dot product of L2-normalized query vector against pre-indexed sparse document-term matrix.

### 2.2. Baseline 2: Okapi BM25 with Financial Inverted Index
- **Algorithm:** Okapi BM25 with Robertson-Spärck Jones IDF and non-negative IDF flooring.
- **Hyperparameters:** $k_1 = 1.5$, $b = 0.75$, $\epsilon = 0.25$.
- **Indexing:** Inverted index mapping financial tokens to document IDs and term frequencies.
- **Query Processing:** Dynamic vector accumulator scoring matching documents with document length normalization.

---

## 3. Measured Retrieval Results

Because coverage is 100.0% (all 150 questions covered by indexed documents), the eligible subset metrics and total benchmark denominator metrics are identical ($N = 150$).

### 3.1. Overall Performance Comparison

| Metric | Word TF-IDF (Cosine) | Okapi BM25 | Absolute Gain | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Gold Document Recall@1** | 24.67% (37 / 150) | **35.33%** (53 / 150) | +10.66% | +43.2% |
| **Gold Document Recall@3** | 43.33% (65 / 150) | **51.33%** (77 / 150) | +8.00% | +18.5% |
| **Gold Document Recall@5** | 54.67% (82 / 150) | **59.33%** (89 / 150) | +4.66% | +8.5% |
| **Gold Evidence Page Recall@1** | 4.00% (6 / 150) | **8.67%** (13 / 150) | +4.67% | +116.8% |
| **Gold Evidence Page Recall@3** | 7.33% (11 / 150) | **10.00%** (15 / 150) | +2.67% | +36.4% |
| **Gold Evidence Page Recall@5** | 8.67% (13 / 150) | **12.67%** (19 / 150) | +4.00% | +46.1% |
| **Gold Evidence Page MRR** | 0.0589 | **0.0986** | +0.0397 | +67.4% |
| **Mean Query Latency (CPU)** | 151.02 ms | **26.06 ms** | -124.96 ms | **5.8x faster** |

---

## 4. Qualitative Error Analysis

To understand why pure lexical retrieval fails on ~87% of financial questions at the page level, five error cases from the BM25 evaluation were manually inspected against the ground truth annotations:

### Error Case 1: Cross-Temporal Boilerplate Distraction
- **Question ID:** `financebench_id_03029`
- **Question:** *"What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."*
- **Gold Target:** `3M_2018_10K`, Page 59 (Consolidated Statement of Cash Flows).
- **Gold Doc in Top-5:** Yes.
- **Retrieved Top 3:**
  1. `3M_2023Q2_10Q` (Page 69, Score: 28.97) — MD&A Financial Condition & Liquidity.
  2. `3M_2022_10K` (Page 36, Score: 28.20) — MD&A Financial Condition & Liquidity.
  3. `3M_2018_10K` (Page 82, Score: 27.26) — Accumulated Other Comprehensive Income.
- **Root Cause:** Standard SEC MD&A liquidity sections repeatedly discuss "free cash flow capability", "capital expenditures", and "USD millions" across successive annual reports. These dense prose discussions accumulate higher BM25 term frequency scores than the actual numeric table on Page 59, pushing the gold cash flow statement out of the top 5.

### Error Case 2: Query Phrasing Noise & Persona Terminology
- **Question ID:** `financebench_id_04672`
- **Question:** *"Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions."*
- **Gold Target:** `3M_2018_10K`, Page 57 (Consolidated Balance Sheet).
- **Gold Doc in Top-5:** Yes.
- **Retrieved Top 3:**
  1. `3M_2018_10K` (Page 98, Score: 31.09) — Note 14: Equities Mandates.
  2. `3M_2023Q2_10Q` (Page 17, Score: 26.23) — Note 8: Income Taxes.
  3. `3M_2022_10K` (Page 90, Score: 25.55) — Derivative Instruments.
- **Root Cause:** The conversational prompt prefix *"Assume that you are a public equities analyst"* heavily weighted terms like "equities" and "analyst". Consequently, Page 98 (*"Equities consist primarily of mandates in public equity securities..."*) dominated BM25 scoring over the clean balance sheet showing Net Property, Plant and Equipment on Page 57.

### Error Case 3: Implicit Conceptual Queries (Capital Intensity)
- **Question ID:** `financebench_id_00499`
- **Question:** *"Is 3M a capital-intensive business based on FY2022 data?"*
- **Gold Target:** `3M_2022_10K`, Pages 47, 49, 51 (Income Statement, Balance Sheet, Cash Flow Statement).
- **Gold Doc in Top-5:** Yes.
- **Retrieved Top 3:**
  1. `3M_2022_10K` (Page 57, Score: 14.63) — Note on Fair Value Measurements.
  2. `3M_2018_10K` (Page 68, Score: 14.41) — Operating Income Footnotes.
  3. `3M_2022_10K` (Page 60, Score: 13.32) — Acquisitions and Divestitures.
- **Root Cause:** The query asks a high-level financial reasoning question ("capital-intensive"). Financial statements never explicitly label themselves as "capital-intensive"; an analyst computes capex / revenue or depreciation / total assets. Lexical BM25 has zero semantic awareness of financial formulas, matching incidental footnote terms instead of primary financial statements.

### Error Case 4: Cross-Company Lexical Overlap
- **Question ID:** `financebench_id_01226`
- **Question:** *"What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why."*
- **Gold Target:** `3M_2022_10K`, Page 26 (MD&A Results of Operations).
- **Gold Doc in Top-5:** **No** (Complete document miss).
- **Retrieved Top 3:**
  1. `COCACOLA_2022_10K` (Page 47, Score: 24.01) — Operating Margin & Channel Mix.
  2. `3M_2023Q2_10Q` (Page 56, Score: 23.95) — Non-GAAP Adjustments.
  3. `COCACOLA_2021_10K` (Page 46, Score: 23.94) — Operating Margin by Segment.
- **Root Cause:** Because Coca-Cola's 10-K devotes extensive paragraphs to detailed segment operating margin discussions, its pages matched terms like "operating margin change", "operating segments", and "favorable channel mix" with higher density than 3M's Page 26. Without document-level metadata filtering or entity resolution, cross-document lexical pollution occurs.

### Error Case 5: Temporal Lookahead & Quarter Mismatch
- **Question ID:** `financebench_id_01865`
- **Question:** *"If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?"*
- **Gold Target:** `3M_2022_10K`, Page 24 (Segment Performance Summary).
- **Gold Doc in Top-5:** Yes.
- **Retrieved Top 3:**
  1. `3M_2023Q2_10Q` (Page 67, Score: 27.96) — Health Care Business Q2 results.
  2. `3M_2023Q2_10Q` (Page 66, Score: 23.98) — Transportation & Electronics Q2 results.
  3. `3M_2022_10K` (Page 31, Score: 23.23) — Health Care Business segment details.
- **Root Cause:** Both the 2023 Q2 10-Q and the 2022 10-K describe segment growth and exclude special items. The quarterly report pages had higher term density for "Sales change analysis", ranking above the annual 10-K page.

---

## 5. Artifact Provenance & Environment

- **Python Version:** `3.14.5`
- **PyPDF Version:** `6.21.1`
- **Cryptography Version:** `50.0.2`
- **Scikit-Learn Version:** `1.9.1`
- **NumPy Version:** `2.5.4`
- **Hardware Profile:** AMD64 CPU, 16 GB RAM (Zero GPU/CUDA utilized)
- **Derived Artifacts Generated:**
  - [`artifacts/financebench_retrieval/metrics.json`](file:///c:/Users/Aayan/Desktop/Project/NLP/artifacts/financebench_retrieval/metrics.json)
  - [`artifacts/financebench_retrieval/config.json`](file:///c:/Users/Aayan/Desktop/Project/NLP/artifacts/financebench_retrieval/config.json)
  - [`configs/datasets/financebench.yaml`](file:///c:/Users/Aayan/Desktop/Project/NLP/configs/datasets/financebench.yaml)

---

## 6. Strategic Architecture Implications for Future Checkpoints

The empirical findings from Checkpoint 04 establish why standard off-the-shelf naive RAG fails on corporate financial filings:
1. **Document Routing is Mandatory:** Document-level filtering (matching company ticker and filing period before searching pages) would immediately boost recall by eliminating cross-company and cross-temporal distractions.
2. **Table-Aware Chunking & Semantic Search:** Financial statements (Balance Sheet, Cash Flow, Income Statement) require specialized semantic headers and structural markdown tables rather than flat page text.
3. **Corrective RAG (CRAG) Engine:** When lexical or embedding retrieval produces candidate pages that lack the requested financial metrics, a corrective fallback mechanism (evaluating retrieval confidence and routing to table indices or footnote searches) is required.

---
*End of FinVeritas FinanceBench Evidence Retrieval Report.*
