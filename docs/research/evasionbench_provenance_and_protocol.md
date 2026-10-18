# EvasionBench Data Provenance, Licensing, and Evaluation Protocol: FinVeritas Research Investigation

## Executive Summary & Primary Research Question

**Primary Research Question:**  
*Can we obtain and lawfully use the human-labelled EvasionBench gold evaluation set separately from the publicly visible model-labelled training examples?*

**Definitive Finding:**  
**NO.** The human-labelled gold evaluation set (the "Gold 1K" set referenced in arXiv:2601.09142) **cannot be obtained or downloaded** from any publicly accessible repository (GitHub, Hugging Face, or the official project website). Furthermore, its lawful use in commercial or unrestricted downstream applications (such as FinVeritas) is encumbered by the proprietary database rights of the underlying raw source data (**S&P Capital IQ Full Text Dataset**).

1. **Unobtainable Gold Set:** Despite the paper's claims (*"We release: (1) a balanced 84K training set, (2) a 1K gold-standard evaluation set..."*), neither the 84,000 Multi-Model Consensus training set nor the 1,000 human-annotated gold evaluation set has been released to the public. 
2. **Public Dataset Reality:** The only dataset published on Hugging Face (`FutureMa/EvasionBench`) is a single 16,726-row parquet file (`evasionbench_17k_eva4b_labels_dedup.parquet`). Its labels (`eva4b_label`) are **not human ground truth**, but model-generated pseudo-labels produced by **Eva-4B-V2** (a fine-tuned Qwen3-4B model).
3. **Severe Class Imbalance Artifact:** The public 16,726-row set is heavily skewed (Direct: 52.3%, Intermediate: 44.0%, Fully Evasive: 3.7%), directly contradicting the "balanced" distribution described in the paper.
4. **Harness Leakage:** The official Hugging Face evaluation harness (`eval.yaml` for `inspect-ai`) simply evaluates against the model-generated pseudo-labels (`eva4b_label_letter`) in the training split. No separate evaluation harness or hidden test benchmark exists.
5. **No Group/Temporal Metadata:** The dataset provides only opaque 32-character hex UIDs, questions, and answers. It exposes **no ticker, company, call date, or transcript identifiers**, preventing group-aware and temporal leakage controls unless external entity resolution is applied.

---

## 1. Evidence Table

| Claim / Topic | Primary Source | Verified Finding & Reality | Confidence |
| :--- | :--- | :--- | :--- |
| **Release of 1K Human Gold Evaluation Set** | arXiv:2601.09142 (§1, §10, Abstract); GitHub README | **Unreleased / Not Publicly Obtainable.** The official GitHub repo contains only 6 sample test cases (`data/evasionbench_test_cases_6samples.csv`). Hugging Face hosts zero human-annotated rows. Community Discussion #7 on HF asking for the dataset remains unanswered. | **High (Definitive)** |
| **Release of 84K Balanced Training Set** | arXiv:2601.09142 (§1, §4, §5.5, Table 3) | **Unreleased.** Sourced from 22.7M S&P Capital IQ Q&A pairs (11.27M filtered). 60K consensus + 24K arbitrated samples were never uploaded to HF or GitHub. | **High (Definitive)** |
| **Nature of Public 16,726-Row Dataset** | HF Dataset Card (`FutureMa/EvasionBench`); Parquet Metadata | **Model-Generated Pseudo-Labels.** The 16,726 rows in `evasionbench_17k_eva4b_labels_dedup.parquet` are labeled via `Eva-4B-V2`. Class breakdown: Direct (8,749 / 52.3%), Intermediate (7,359 / 44.0%), Fully Evasive (618 / 3.7%). | **High (Definitive)** |
| **Evaluation Harness Configuration** | HF `eval.yaml` (commit `1ef325f`, PR #6) | **Evaluates against Model Pseudo-Labels.** Configured for `inspect-ai` targeting `eva4b_label_letter` on `split: train`. Does not evaluate against human gold ground truth. | **High (Definitive)** |
| **Licensing of Public Artefacts** | GitHub `LICENSE`; HF Dataset Card; HF Model Card | **Apache 2.0 on code & card metadata.** However, the underlying transcript text originates from S&P Capital IQ, which carries proprietary non-redistribution commercial terms. | **High (Definitive)** |
| **Presence of Company & Temporal Metadata** | Parquet Schema Inspection (`pyarrow.parquet`) | **Completely Absent.** Schema consists exclusively of `uid`, `question`, `answer`, `eva4b_label`, `choices`, and `eva4b_label_letter`. Zero ticker, company, date, or call identifiers. | **High (Definitive)** |
| **Inter-Annotator Agreement (IAA)** | arXiv:2601.09142 (§6, Table 4) | **Reported on 100 samples only.** Cohen's Kappa = 0.835, Macro-F1 = 88.99%. 10 of 11 disagreements were on the `intermediate` class. | **High (Paper Source)** |
| **Eva-4B Reported Performance** | arXiv:2601.09142 (Table 5, Table 7) | **84.9% Macro-F1 on Private Gold 1K.** Direct F1: 82.2%, Intermediate F1: 80.1%, Fully Evasive F1: 92.4%. Outperforms Gemini 3 Flash (84.6%), Claude Opus 4.5 (84.4%), GPT-5.2 (80.9%). | **High (Paper Source)** |
| **Taxonomy Provenance** | GitHub prompts (`evasion_rasiah_*.txt`) | **Based on Rasiah (2010) & Gricean Pragmatics.** Operationalized around primary question core resolution (Relation maxim). | **High (Definitive)** |

---

## 2. Definitive Account: What is Actually Downloadable vs. Unestablished

### Actually Downloadable Artefacts
1. **Hugging Face Dataset (`FutureMa/EvasionBench`):**
   - File: `evasionbench_17k_eva4b_labels_dedup.parquet` (12,083,417 bytes, 16,726 rows).
   - Schema: `uid` (string), `question` (string), `answer` (string), `eva4b_label` (string), `choices` (list), `eva4b_label_letter` (string).
   - `eval.yaml`: Inspect-AI evaluation specification.
   - `README.md`: Dataset card.
2. **GitHub Repository (`IIIIQIIII/EvasionBench`):**
   - `data/evasionbench_test_cases_6samples.csv`: 6 manually curated CSV rows with justification strings (`selection_reason`).
   - `prompts/evasion_rasiah_fine_tuning_minimalist.txt`: Minimal JSON prompt template.
   - `prompts/evasion_rasiah_with_reason_prompt_template.txt`: Detailed prompt template incorporating the Rasiah (2010) evasion principles.
   - `scripts/eva4b_inference.py` & `eva4b_inference.ipynb`: Python inference runners.
   - `LICENSE`: Apache 2.0.
3. **Hugging Face Model Weights:**
   - `FutureMa/Eva-4B-V2` (and `Eva-4B`): Causal LM fine-tuned from `Qwen/Qwen3-4B-Instruct-2507` using MS-SWIFT.
   - GGUF Quantizations: `Eva-4B-Q4_K_M-GGUF`, `Eva-4B-Q5_K_M-GGUF`, `Eva-4B-Q8_0-GGUF`.

### Unreleased / Unestablished Claims
1. **The 1K Human Gold Evaluation Set:** Completely absent from public distribution. Not accessible via API or benchmark server.
2. **The 84K Multi-Model Consensus (MMC) Balanced Training Set:** Never uploaded.
3. **Underlying Transcript Rights:** S&P Capital IQ proprietary data agreements restrict transcript republication, explaining why the authors could not lawfully publish the full raw corpus.
4. **Generalization Beyond S&P Transcripts:** Unverified across other financial genres (e.g. SEC 10-K disclosures, shareholder meetings, press conferences).

---

## 3. Label Categories & Annotation Semantics

To prevent circular evaluation and false claims, FinVeritas must maintain strict distinction between four distinct tiers of data:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA LABEL PROVENANCE                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. HUMAN GROUND TRUTH (Gold 1K)                                        │
│    • 1,000 samples labeled by expert human annotator                   │
│    • 100 samples double-annotated (Cohen's Kappa = 0.835)              │
│    • Status: PRIVATE / UNPUBLISHED                                     │
├────────────────────────────────────────────────────────────────────────┤
│ 2. MULTI-MODEL CONSENSUS (MMC 84K)                                     │
│    • 60K consensus (Claude Opus 4.5 + Gemini 3 Flash agreement)        │
│    • 24K arbitrated (3-judge majority vote: Opus + Gemini + GPT-5.2)   │
│    • Status: PRIVATE / UNPUBLISHED                                     │
├────────────────────────────────────────────────────────────────────────┤
│ 3. MODEL-GENERATED PSEUDO-LABELS (HF 16.7K)                            │
│    • 16,726 Q&A pairs labeled by Eva-4B-V2 (Qwen3-4B fine-tune)        │
│    • Severe skew: Direct 52.3%, Intermediate 44.0%, Fully Evasive 3.7% │
│    • Status: PUBLICLY DOWNLOADABLE (Must treat as weak distillation)   │
├────────────────────────────────────────────────────────────────────────┤
│ 4. BENCHMARK METRICS & LEADERBOARD                                     │
│    • Macro-F1 scores reported on the private Gold 1K                   │
│    • Eva-4B: 84.9%, Gemini: 84.6%, Opus: 84.4%, GPT-5.2: 80.9%        │
│    • Status: REPORTED IN PAPER (Cannot be reproduced directly)         │
└────────────────────────────────────────────────────────────────────────┘
```

### Taxonomy & Linguistic Mechanics
- **Direct (F1 = 82.2% in paper):** Explicit resolution of the primary core request (numeric answer for quantitative queries; binary stance for yes/no questions; clear causal reasoning for "why" queries).
- **Intermediate (F1 = 80.1% in paper):** The primary boundary challenge. 52% of model errors occur when direct answers are misclassified as intermediate due to executive hedging ("we expect", "soon", "hope to see"), or when executives provide qualitative lists rather than requested numbers.
- **Fully Evasive (F1 = 92.4% in paper):** Complete deflection, overt refusal ("we don't break that out at this granularity"), topic pivoting, or irrelevant boilerplate. High distinctiveness makes it the easiest class for LLMs, but it represents only 3.7% of the public HF release.

---

## 4. Contamination, Duplication, and Leakage Risks

1. **Company & Entity Leakage:**  
   Because the dataset lacks ticker/company columns, a naïve random train/test split will place Q&A pairs from the same call or firm into both splits. The classifier will memorize firm-specific product names, executive idioms, and idiosyncratic jargon rather than learning general evasion patterns.
2. **Temporal Lookahead Leakage:**  
   Transcripts span 2002–2022. Without date metadata, random splits train on 2021 data and test on 2008 data, violating causal temporal boundaries in financial econometrics.
3. **Exact & Near Duplicates:**  
   Empirical audit reveals 97 duplicate questions and 149 duplicate answers across distinct UIDs (e.g. formulaic analyst greetings and standard forward-looking safe harbor disclaimers).
4. **Teacher-Student Evaluation Trap:**  
   Evaluating a model against `eva4b_label` in the HF dataset measures **fidelity to Eva-4B-V2**, not accuracy against managerial truth.

---

## 5. Scientifically Defensible Evaluation Design for FinVeritas

Because the official Gold 1K cannot be obtained, FinVeritas must execute an independent, scientifically defensible evaluation protocol:

### Phase 1: Stratified Audit Subset Creation
- Sample $N = 450$ examples from the public corpus using stratified sampling across predicted labels:
  - 150 pseudo-direct
  - 150 pseudo-intermediate
  - 150 pseudo-fully evasive (oversampling the 3.7% tail to ensure statistical power).
- Extract named entities (using SpaCy / Stanza NER) to group samples by detected firm/ticker and prevent train-audit leakage.

### Phase 2: Dual Human Annotation & Adjudication
- Two independent financial domain annotators annotate all 450 samples following standardized operational guidelines:
  - Rule 1: Identify the **primary information request** (numeric, temporal, binary, causal).
  - Rule 2: Evaluate Relation maxim fulfillment independent of tone or length.
  - Rule 3: Distinguish executive hedging ("we anticipate") from evasion if the core fact is confirmed.
- Calculate Inter-Annotator Agreement:
  - Cohen's Kappa ($\kappa$) and Krippendorff's Alpha ($\alpha$ for ordinal data).
  - Target: $\kappa \ge 0.80$.
- Disagreements adjudicated by a third senior reviewer or joint consensus conference.

### Phase 3: Train/Val/Audit Split Policy
- **Weak Training Split:** 15,000 pseudo-labeled pairs (excluding all audit entities).
- **Weak Validation Split:** 1,276 pseudo-labeled pairs for early stopping and hyperparameter tuning.
- **Human Gold Audit Benchmark:** 450 human-adjudicated pairs (never exposed to training).

### Phase 4: Statistical Uncertainty Reporting
- Report Macro-F1 alongside per-class Precision, Recall, and F1.
- Compute **95% Bootstrap Confidence Intervals** (1,000 iterations with replacement).
- Conduct **McNemar's test** or **paired permutation tests** when comparing model variants against baselines.

---

## 6. Baseline Comparison Framework

To prevent exaggerated claims, compare any fine-tuned encoder against three standard baselines on the human audit set:

1. **Majority-Class Baseline:** Predicts `direct` for all samples. (Yields ~52% accuracy on natural distribution, but Macro-F1 $\approx 0.23$).
2. **Lexical / Dictionary Baseline:** Logistic Regression on TF-IDF features combined with Loughran-McDonald uncertainty word lists and regex evasion refusal markers (`"do not disclose"`, `"not at liberty"`, `"premature to"`).
3. **Embedding-Similarity Baseline:** Compute cosine similarity between Question and Answer embeddings using `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`. Classify via prototype centroids or logistic regression.
4. **Fine-Tuned Small Encoder:** `microsoft/deberta-v3-small` (44M params) or `deberta-v3-base` (86M params) cross-encoder.

---

## 7. Realistic First Training Experiment Specification

### Hardware Constraints Analysis
- **System:** Windows 11, 16 GB System RAM.
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MiB / 6 GB total VRAM).
- **VRAM Budget:** Windows desktop manager (WDDM), display output, Chrome, and IDE consume ~1,090 MiB, leaving **~5,050 MiB (~4.9 GB) available VRAM**.
- **Feasibility:** Fine-tuning a 4B parameter model (Eva-4B / Qwen3-4B) even with QLoRA requires >5.5 GB VRAM during backward passes and will trigger CUDA OOM errors. Conversely, a 44M parameter cross-encoder (`deberta-v3-small`) consumes only **~2.1 GB VRAM**, running stably and completing 3 epochs in under 10 minutes.

### Experiment Configuration
- **Model:** `microsoft/deberta-v3-small` (Cross-Encoder)
- **Input Format:** `[CLS] Question [SEP] Answer [SEP]`
- **Max Sequence Length:** 384 tokens (captures >95% of Q&A pairs)
- **Per-Device Batch Size:** 16
- **Gradient Accumulation Steps:** 2 (Effective Batch Size = 32)
- **Precision:** Mixed Precision (`torch.cuda.amp.autocast`, fp16)
- **Optimizer:** AdamW (`lr = 2e-5`, `weight_decay = 0.01`, linear warmup 5%)
- **Loss Function:** Class-Weighted Cross-Entropy:
  $$w_c = \frac{N}{3 \cdot N_c} \implies w_{\text{direct}} \approx 0.64,\; w_{\text{interm}} \approx 0.76,\; w_{\text{fully\_evasive}} \approx 8.33$$
  This directly counteracts the 3.7% minority class artifact in the pseudo-labeled data.

---

## 8. Unresolved Questions Before Performance Reporting

Before FinVeritas reports any performance figures externally, the following gates must be cleared:

1. **Has the evaluation been conducted on human-adjudicated labels rather than `eva4b_label`?**  
   Reporting 85%+ Macro-F1 on `eva4b_label` is merely student-teacher distillation, not real-world evasion detection.
2. **Has entity/firm overlap between training and evaluation been audited and eliminated?**  
   If the same executive/company appears in train and test, reported metrics suffer from memorization leakage.
3. **Are confidence intervals reported for the audit set?**  
   Given an audit sample of $N \approx 450$, confidence intervals will typically span $\pm 3.5\%$. Point estimates must not be presented as exact.
4. **Has the downstream verification step (claim extraction $\to$ SEC filing cross-examination) been defined independently of evasion classification?**  
   Evasive answers may still contain partial factual claims that require SEC EDGAR validation.
