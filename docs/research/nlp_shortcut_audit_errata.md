# FinVeritas: Scientific Interpretation Errata & Corrections (Checkpoint 06A)

- **Date:** 2026-10-19
- **Checkpoint:** Checkpoint 06A (`checkpoint/06a-nlp-report-corrections`)
- **Target Document:** [`docs/research/nlp_shortcut_audit.md`](file:///docs/research/nlp_shortcut_audit.md)
- **Base Commit:** `d453cea` (`checkpoint/06-nlp-shortcut-audit`)
- **Status:** Verified & Ratified

---

## 1. Motivation & Scope

Following an internal scientific review of Checkpoint 06 (`checkpoint/06-nlp-shortcut-audit`), this document specifies and ratifies necessary scientific interpretation corrections to the original NLP Shortcut Learning and Robustness Audit report ([`docs/research/nlp_shortcut_audit.md`](file:///docs/research/nlp_shortcut_audit.md)).

All underlying empirical measurements, saved metrics artifacts ([`artifacts/nlp_shortcut_audit/metrics.json`](file:///artifacts/nlp_shortcut_audit/metrics.json)), split datasets, and unit tests remain unaltered and scientifically valid. No additional GPU training or metric fabrication was performed. The revisions strictly address scientific overgeneralization, inaccurate causal claims, and proper contextualization of pseudo-label evaluations.

---

## 2. Summary of Corrections

### 2.1 Correction 1: Input-Ablation Interpretation & Trade-Off

| Area | Original Phrasing / Claim | Corrected Scientific Interpretation |
| :--- | :--- | :--- |
| **Question-Only Minority Collapse** | *"The Question Contains Zero Evasion Signal: ... questions alone cannot predict whether an executive will be evasive."* | Question-only DeBERTa obtained 0 recall and 0 F1 on `fully_evasive` in the evaluated configuration (0 out of 124 instances detected). This finding reflects a failure mode under this specific model architecture, fine-tuning setup, and evaluation cohort. **It does not establish that questions contain zero predictive information across all classes or models.** The question-only model achieved 64.45% F1 on `direct` and 61.35% F1 on `intermediate` (61.81% accuracy), demonstrating non-trivial predictive structure across majority classes. |
| **Answer-Only vs. Question+Answer** | *"The question text acts as a distractor during fine-tuning, introducing gradient noise without contributing discriminative signal for evasion classification."* | Question text **must not be described as a universal distractor**. Answer-only achieved higher overall Macro-F1 than question+answer on the existing held-out test set (0.6251 vs. 0.5959, $+0.0292$, $p = 0.008$). However, Question+Answer achieved substantially higher fully-evasive recall: **62.90% (78/124)** compared with **49.19% (61/124)** for Answer-Only (+13.71% recall advantage). Answer-only and question+answer therefore exhibit an **empirical trade-off between aggregate classification performance and evasive-example sensitivity**. Claims are strictly bounded to this model, training setup, and evaluation population. |
| **Target Nature & Human Validation** | Implied general claim about real-world managerial evasion. | All reported classification targets (`eva4b_label`) are **model-generated pseudo-labels** produced by Eva-4B-V2 (arXiv:2601.09142), not human consensus annotations. High agreement reflects alignment with Eva-4B-V2's scoring behavior, **not independent human validation of conversational evasion**. |

### 2.2 Correction 2: Answer-Template Leakage & Split Comparability

| Area | Original Phrasing / Claim | Corrected Scientific Interpretation |
| :--- | :--- | :--- |
| **Causal Memorization Claim** | *"Memorization of formulaic corporate answer templates accounts for approximately 10.9 F1 points of apparent evasion detection capability."* | The claim that approximately 10.9 F1 points of apparent capability were **directly caused by memorization has been removed and qualified**. The standard question-disjoint split ($N=3,346$) and strict answer-disjoint split ($N=3,346$) constitute **mathematically distinct evaluation populations** with different sample subsets. While the observed drop in Fully-Evasive F1 (0.5065 down to 0.3977) is consistent with a shortcut-learning concern (potential exploitation of recurring answer phrasing), this cross-population comparison **does not isolate the causal contribution of memorization from sampling and population distribution shifts**. |
| **Retention of Both Benchmarks** | Direct comparison between non-identical populations. | Both original results, full confusion matrices, and split descriptions are preserved. Neither evaluation population is substituted for the other, nor is the observed degradation concealed. Future model development should report performance across **both benchmarks**, treating the strict answer-disjoint split as a stress test for shortcut vulnerability rather than an isolated causal metric of memorization. |

---

## 3. Preserved Empirical Results

The empirical results from Checkpoint 06 remain intact and reproducible:

1. **Input Ablation Metrics (Held-Out Test Set, $N=3,346$):**
   - Question-Only: Accuracy 61.81%, Macro-F1 0.4193, Fully-Evasive F1 0.0000 (Recall: 0.00%, Precision: 0.00%)
   - Answer-Only: Accuracy 67.51%, Macro-F1 0.6251, Fully-Evasive F1 0.5169 (Recall: 49.19%, Precision: 54.46%)
   - Question+Answer: Accuracy 63.87%, Macro-F1 0.5959, Fully-Evasive F1 0.5065 (Recall: 62.90%, Precision: 42.39%)
   - Paired Bootstrap Difference (Answer-Only vs. QA): $\Delta$ Macro-F1 = $+0.0289$, 95% CI `[+0.0044, +0.0510]`, $p = 0.008$
2. **Answer-Disjoint Metrics (Held-Out Test Set, $N=3,346$):**
   - QA Model on Strict Answer-Disjoint: Accuracy 69.73%, Macro-F1 0.5935, Fully-Evasive F1 0.3977 (Recall: 28.23%, Precision: 67.31%)
   - Group Overlap: Zero answer overlap between train, val, and test.

---

## 4. Verification & Testing

Integrity check unit tests were added in [`tests/unit/test_nlp_shortcut_audit.py`](file:///tests/unit/test_nlp_shortcut_audit.py) verifying:
1. Exact match between documented report metrics and the frozen [`artifacts/nlp_shortcut_audit/metrics.json`](file:///artifacts/nlp_shortcut_audit/metrics.json) file.
2. Verified absence of discredited causal memorization and universal distractor claims in the research report.
3. Verification of dual-split population notices and pseudo-label governance disclaimers.
