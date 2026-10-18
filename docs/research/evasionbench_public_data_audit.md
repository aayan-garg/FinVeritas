# EvasionBench Public Data Audit & Provenance Verification

- **Date:** 2026-10-18
- **Checkpoint:** Checkpoint 01 (EvasionBench Dataset Audit)
- **Status:** Completed & Grounded in Directly Observed Artifacts
- **Author:** FinVeritas Engineering & ML Data Team

---

## Executive Summary & Core Verdict

This document records the empirical audit of the publicly accessible dataset artifacts for **EvasionBench** (arXiv:2601.09142), conducted as part of the FinVeritas project bootstrap.

### Core Verdict
1. **The 1,000-sample human gold evaluation set and the 84,000-sample multi-model consensus (MMC) training set advertised in arXiv:2601.09142 are NOT publicly accessible.** They cannot be downloaded from Hugging Face, GitHub, or any public API without private credentials or author intervention.
2. **The sole publicly downloadable dataset artifact is a 16,726-row Parquet file (`FutureMa/EvasionBench`).**
3. **The public labels (`eva4b_label`) are model-generated pseudo-labels produced by Eva-4B-V2 (a fine-tuned Qwen3-4B model), NOT human annotations.**
4. **The dataset displays severe class imbalance:** `direct` accounts for 52.31%, `intermediate` for 44.00%, and `fully_evasive` accounts for only 3.69% (618 rows).
5. **Group-aware and temporal leakage controls are impossible without external entity resolution:** The dataset exposes zero ticker, company, call date, or transcript identifiers.
6. **Reproducing the published benchmark is currently impossible:** The published 84.9% Macro-F1 metric was evaluated on the private human Gold 1K benchmark, which is unreleased.

---

## 1. Upstream Claims vs. Directly Observed Facts

| Topic | Upstream Claim (Paper / GitHub) | Directly Observed Reality (Empirical Audit) | Classification |
| :--- | :--- | :--- | :--- |
| **84K Training Set** | "We release a balanced 84K training set..." (arXiv:2601.09142 §1, Table 3) | **Absent.** Zero 84K dataset files exist on Hugging Face or GitHub. | Confirmed Fact (Absence) |
| **1K Human Gold Benchmark** | "We release a 1K gold-standard evaluation set verified by human experts..." (arXiv:2601.09142 §1, §6) | **Absent.** GitHub repo contains only 6 sample test cases (`data/evasionbench_test_cases_6samples.csv`). Community discussion #7 on HF asking for the dataset remains unanswered. | Confirmed Fact (Absence) |
| **Public Parquet Dataset** | 16,726 samples deduplicated from earnings call transcripts (`FutureMa/EvasionBench`) | **Present.** Exactly 1 Parquet file (`evasionbench_17k_eva4b_labels_dedup.parquet`) of 12,083,417 bytes (SHA-256: `8ca4194c00cb037e9d719d45a569935aaabb31c3f781f03c1fcb9c7db875508b`). | Confirmed Fact (Observed) |
| **Label Source** | Model-generated pseudo-labels via Eva-4B-V2 (HF dataset card description) | **Verified.** `eva4b_label` contains pseudo-labels from Eva-4B-V2 distillation, not ground-truth human annotations. | Confirmed Fact (Observed) |
| **Dataset Balance** | Paper claims training sets were filtered to balanced class distributions. | **Contradicted.** Public data is heavily skewed: 52.31% direct, 44.00% intermediate, 3.69% fully evasive. | Confirmed Fact (Observed) |
| **Evaluation Harness** | Hugging Face `eval.yaml` (Inspect-AI) | Evaluates against `eva4b_label_letter` on `split: train`. Evaluates model outputs against model pseudo-labels. | Confirmed Fact (Observed) |
| **Repository License** | Apache-2.0 in GitHub and Hugging Face card metadata | Repository code and metadata are licensed under Apache-2.0. However, underlying transcript texts originate from commercial sources (S&P Capital IQ). | Confirmed Fact & IP Risk |

---

## 2. Technical Inspection & Schema Verification

### 2.1. File Metadata & Checksums
- **Local Path:** `data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet`
- **Source URL:** `https://huggingface.co/datasets/FutureMa/EvasionBench/resolve/main/evasionbench_17k_eva4b_labels_dedup.parquet`
- **Observed HF Commit / Revision:** `4b91247a823ca046d9f1d936be7dc2fe0735ad08`
- **File Size:** `12,083,417 bytes` (11.52 MB)
- **SHA-256 Checksum:** `8ca4194c00cb037e9d719d45a569935aaabb31c3f781f03c1fcb9c7db875508b`
- **Row Count:** `16,726`
- **Row Group Count:** `1`

### 2.2. Parquet Schema
The physical Parquet schema inspected via `pyarrow.parquet` contains exactly six fields:

```text
Field Name            Arrow Type
-----------------------------------------------------------
uid                   string
question              string
answer                string
eva4b_label           string
choices               list<element: string>
eva4b_label_letter    string
```

### 2.3. Label Distribution & Severe Minority Skew
Class distribution across the 16,726 rows:

| Label (`eva4b_label`) | Sample Count | Percentage | Linguistic Category |
| :--- | :--- | :--- | :--- |
| **`direct`** | 8,749 | **52.31%** | Answer resolves the primary question core. |
| **`intermediate`** | 7,359 | **44.00%** | Partial answer, qualitative response, or hedged statement. |
| **`fully_evasive`** | 618 | **3.69%** | Total deflection, refusal to answer, topic shift, or boilerplate. |
| **Total** | **16,726** | **100.00%** | |

> [!WARNING]
> **Extreme Minority Imbalance:** `fully_evasive` represents only 3.69% of the dataset. Naive cross-entropy loss will cause standard classifiers to predict either `direct` or `intermediate`, effectively ignoring `fully_evasive`. Balanced loss weighting (or focal loss) is mathematically mandatory.

### 2.4. Null, Blank, and Integrity Checks
- **Null Values:** `0` across all columns (`uid`, `question`, `answer`, `eva4b_label`, `choices`, `eva4b_label_letter`).
- **Whitespace / Blank Strings:** `0` across all columns.
- **Letter Mapping Validation:** The mapping between `choices`, `eva4b_label_letter` (A, B, C), and `eva4b_label` was evaluated across all 16,726 rows:
  - Total Evaluated: `16,726`
  - Mismatches: `0`
  - Validation Status: **100% Consistent.**

### 2.5. Duplicate Statistics
- **Total Rows:** `16,726`
- **Unique UIDs:** `16,726` (0 duplicate UIDs)
- **Unique Questions:** `16,629` (**97 duplicate questions**)
- **Unique Answers:** `16,577` (**149 duplicate answers**)
- **Unique Question-Answer Pairs:** `16,726` (0 duplicate QA pairs)

**Inference:** While QA pairs are distinct, the presence of 97 repeated questions and 149 repeated answers indicates that common executive phrases (e.g., boilerplate disclaimers or repetitive analyst questions) recur across different call segments.

### 2.6. Character-Length Distributions
Character lengths evaluated across all 16,726 samples:

| Text Field | Minimum | Median | 95th Percentile (p95) | Maximum |
| :--- | :--- | :--- | :--- | :--- |
| **Question** | 16 chars | 315.0 chars | 694.0 chars | 2,609 chars |
| **Answer** | 7 chars | 739.0 chars | 2,308.0 chars | 8,601 chars |

**Tokenization Memory Implications:**
- At an average of ~4 characters per token, the median QA pair requires $\approx \frac{315 + 739}{4} \approx 263$ tokens.
- At the 95th percentile, a QA pair requires $\approx \frac{694 + 2308}{4} \approx 750$ tokens.
- For a cross-encoder budgeted for 6GB VRAM on Windows (e.g., RTX 3050), a `max_length` of 384 or 512 with batch size 16 will truncate the tails of very long executive monologues while preserving >85% of full exchanges.

---

## 3. Absence of Metadata & Leakage Risks

### 3.1. Absent Group & Temporal Identifiers
The dataset contains **no metadata fields** for:
- `ticker` (Stock symbol)
- `company` (Corporate entity)
- `call_date` / `fiscal_period` (Date and quarter of the earnings call)
- `transcript_id` (Call boundary identifier)

### 3.2. Evaluation Impact: Random vs. Group-Aware Splitting
If random K-fold train/test splits are applied:
- Questions and answers from the **same company or same earnings call** will appear in both the training and test sets.
- Models may memorize firm-specific vocabulary, executive speech patterns, or industry terminology rather than learning general evasion semantics.
- **Unresolved Challenge:** Until entity recognition or external metadata matching is applied to link UIDs back to S&P Capital IQ transcripts, group-aware cross-validation cannot be guaranteed.

---

## 4. Benchmark Reproducibility & Ground-Truth Limitations

### Can we reproduce the published human-gold benchmark?
**NO.** The evidence supporting this conclusion is unequivocal:
1. The paper evaluates all headline models (Table 5: Eva-4B 84.9%, Gemini 84.6%, Opus 84.4%, GPT-5.2 80.9%) against the **1K Gold Set**.
2. The 1K Gold Set is private and unpublished.
3. Hugging Face discussion requests for the benchmark data remain unanswered by the authors.
4. The public Hugging Face evaluation harness (`eval.yaml`) merely runs inference against the training split pseudo-labels.

### Implications for FinVeritas
- We cannot claim that a model trained or evaluated on this public dataset achieves human-grounded accuracy.
- Any model trained on this public dataset is undergoing **knowledge distillation from Eva-4B-V2**, learning its strengths as well as its systemic biases.
- To establish true ground truth, FinVeritas will need an independent human-validated evaluation sample (e.g., 200–500 double-annotated earnings call exchanges) before production deployment.

---

## 5. Licensing, Rights, and Storage Governance

1. **Repository Code & Cards:** Licensed under Apache-2.0.
2. **Underlying Transcript Rights:** As documented in arXiv:2601.09142, the source data was obtained from the **S&P Capital IQ Full Text Dataset**, a commercial proprietary product with explicit redistribution restrictions.
3. **Data Governance Recommendation:**
   - Raw Parquet datasets must **never be committed to Git** or pushed to public GitHub remotes.
   - All dataset paths (`data/raw/`, `data/interim/`, `data/processed/`, `*.parquet`) are strictly enforced in [`.gitignore`](file:///c:/Users/Aayan/Desktop/Project/NLP/.gitignore#L18-L30).
   - Only code, configurations, schemas, and aggregated non-reidentifying statistical audit summaries are version-controlled.

---

## 6. Recommendations & Smallest Justified Next Step

### What the Next NLP Checkpoint Should Implement
Now that the public dataset schema and label provenance are audited and proven to consist of Eva-4B-V2 pseudo-labels with a 3.7% minority class:

1. **Checkpoint 02 (NLP Evasion Baseline Pipeline):**
   - Implement the `finveritas.nlp_evasion` ingestion and dataset preparation pipeline.
   - Implement class-weighted cross-entropy loss: weights must be inversely proportional to class frequencies (weight for `fully_evasive` $\approx 9.0\times$ weight for `direct`).
   - Define a fixed, reproducible local split (e.g., stratified 80/10/10 by `eva4b_label`) with a recorded random seed.
   - Adopt a lightweight baseline architecture (such as DeBERTa-v3-small cross-encoder) that fits within 4GB VRAM under mixed precision.
   - Establish baseline Macro-F1 and per-class Recall metrics against the pseudo-label validation split as a developmental reference.

---

*End of EvasionBench Public Data Audit Report.*
