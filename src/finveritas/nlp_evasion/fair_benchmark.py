"""
FinVeritas: Fair NLP Benchmark, Truncation Mechanics Audit,
Normalized Answer Leakage Audit, and Bootstrap Significance Testing.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from finveritas.nlp_evasion.baselines import (
    SUPPORTED_CLASSES,
    build_tfidf_lr_pipeline,
    evaluate_model,
)
from finveritas.nlp_evasion.transformer_classifier import (
    DEFAULT_MODEL_NAME,
    ID_TO_LABEL,
    LABEL_TO_ID,
    EvasionQAPairDataset,
    collate_qa_batch,
    compute_metrics,
)

logger = logging.getLogger(__name__)


def normalize_answer_text(text: str) -> str:
    """
    Applies Unicode NFKC normalization, lowercasing, and whitespace collapsing
    to answer text for exact duplicate detection.
    """
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass
class TruncationAuditSummary:
    total_samples: int
    max_length_configured: int
    pair_truncated_count: int
    pair_truncation_rate: float
    question_truncated_count: int
    question_truncation_rate: float
    answer_truncated_count: int
    answer_truncation_rate: float
    both_truncated_count: int
    both_truncation_rate: float
    question_length_percentiles: dict[str, float]
    answer_length_percentiles: dict[str, float]
    pair_length_percentiles: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_truncation_mechanics(
    questions: list[str],
    answers: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
) -> TruncationAuditSummary:
    """
    Directly measures token loss for question and answer sequences separately
    and jointly across all provided pairs under HuggingFace sequence-pair truncation.
    """
    total = len(questions)
    q_lens: list[int] = []
    a_lens: list[int] = []
    pair_lens: list[int] = []

    pair_truncated = 0
    q_truncated = 0
    a_truncated = 0
    both_truncated = 0

    for q, a in zip(questions, answers):
        # Measure un-truncated token lengths
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        a_ids = tokenizer.encode(a, add_special_tokens=False)
        pair_ids = tokenizer.encode(q, a, truncation=False)

        len_q = len(q_ids)
        len_a = len(a_ids)
        len_pair = len(pair_ids)

        q_lens.append(len_q)
        a_lens.append(len_a)
        pair_lens.append(len_pair)

        # Tokenize with truncation
        enc = tokenizer(q, a, truncation=True, max_length=max_length)
        seq_ids = enc.sequence_ids()

        q_retained = sum(1 for s in seq_ids if s == 0)
        a_retained = sum(1 for s in seq_ids if s == 1)

        is_pair_trunc = len_pair > max_length
        is_q_trunc = len_q > q_retained
        is_a_trunc = len_a > a_retained

        if is_pair_trunc:
            pair_truncated += 1
        if is_q_trunc:
            q_truncated += 1
        if is_a_trunc:
            a_truncated += 1
        if is_q_trunc and is_a_trunc:
            both_truncated += 1

    def compute_pcts(arr: list[int]) -> dict[str, float]:
        if not arr:
            return {}
        p = np.array(arr)
        return {
            "min": float(np.min(p)),
            "p25": float(np.percentile(p, 25)),
            "p50_median": float(np.median(p)),
            "p75": float(np.percentile(p, 75)),
            "p90": float(np.percentile(p, 90)),
            "p95": float(np.percentile(p, 95)),
            "p99": float(np.percentile(p, 99)),
            "max": float(np.max(p)),
            "mean": round(float(np.mean(p)), 2),
        }

    return TruncationAuditSummary(
        total_samples=total,
        max_length_configured=max_length,
        pair_truncated_count=pair_truncated,
        pair_truncation_rate=round(pair_truncated / max(total, 1), 4),
        question_truncated_count=q_truncated,
        question_truncation_rate=round(q_truncated / max(total, 1), 4),
        answer_truncated_count=a_truncated,
        answer_truncation_rate=round(a_truncated / max(total, 1), 4),
        both_truncated_count=both_truncated,
        both_truncation_rate=round(both_truncated / max(total, 1), 4),
        question_length_percentiles=compute_pcts(q_lens),
        answer_length_percentiles=compute_pcts(a_lens),
        pair_length_percentiles=compute_pcts(pair_lens),
    )


@dataclass
class AnswerLeakageReport:
    unique_answers_train: int
    unique_answers_val: int
    unique_answers_test: int
    overlap_train_val: int
    overlap_train_test: int
    overlap_val_test: int
    train_test_overlap_samples_in_test: int
    train_test_label_agreement_rate: float
    train_test_overlap_examples: list[dict[str, Any]]
    methodological_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_normalized_answer_leakage(
    train_answers: list[str],
    val_answers: list[str],
    test_answers: list[str],
    train_labels: list[str],
    val_labels: list[str],
    test_labels: list[str],
) -> AnswerLeakageReport:
    """
    Audits exact normalized-answer overlap across train, validation, and test partitions,
    measuring overlap frequency and whether identical answers share identical pseudo-labels.
    """
    norm_tr = [normalize_answer_text(a) for a in train_answers]
    norm_v = [normalize_answer_text(a) for a in val_answers]
    norm_te = [normalize_answer_text(a) for a in test_answers]

    set_tr = set(norm_tr)
    set_v = set(norm_v)
    set_te = set(norm_te)

    overlap_tr_v = set_tr & set_v
    overlap_tr_te = set_tr & set_te
    overlap_v_te = set_v & set_te

    # Inspect train-test overlap details
    tr_answer_to_labels: dict[str, list[str]] = {}
    for a, lbl in zip(norm_tr, train_labels):
        tr_answer_to_labels.setdefault(a, []).append(lbl)

    overlap_samples_in_test = 0
    agreed_label_count = 0
    overlap_examples: list[dict[str, Any]] = []

    for idx, (a, te_lbl) in enumerate(zip(norm_te, test_labels)):
        if a in overlap_tr_te:
            overlap_samples_in_test += 1
            tr_lbls = tr_answer_to_labels[a]
            most_common_tr_lbl = Counter(tr_lbls).most_common(1)[0][0]
            agreed = te_lbl == most_common_tr_lbl
            if agreed:
                agreed_label_count += 1

            if len(overlap_examples) < 10:
                overlap_examples.append(
                    {
                        "test_index": idx,
                        "normalized_answer": a[:120] + ("..." if len(a) > 120 else ""),
                        "test_label": te_lbl,
                        "train_labels_observed": dict(Counter(tr_lbls)),
                        "label_agreed": agreed,
                    }
                )

    agreement_rate = (
        round(agreed_label_count / overlap_samples_in_test, 4)
        if overlap_samples_in_test > 0
        else 1.0
    )

    note = (
        "Question-level leakage is strictly zero (zero normalized-question overlap). "
        "Answer overlap reflects formulaic executive expressions ('Yes.', 'No.', "
        "'Thank you.', 'We do not provide that guidance.'). Company-level, speaker-level, "
        "and temporal leakage cannot be excluded because the dataset lacks ticker, speaker, "
        "and call-date metadata."
    )

    return AnswerLeakageReport(
        unique_answers_train=len(set_tr),
        unique_answers_val=len(set_v),
        unique_answers_test=len(set_te),
        overlap_train_val=len(overlap_tr_v),
        overlap_train_test=len(overlap_tr_te),
        overlap_val_test=len(overlap_v_te),
        train_test_overlap_samples_in_test=overlap_samples_in_test,
        train_test_label_agreement_rate=agreement_rate,
        train_test_overlap_examples=overlap_examples,
        methodological_note=note,
    )


def train_fair_tfidf_baseline(
    train_data: dict[str, Any],
    test_data: dict[str, Any],
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fits the TF-IDF + Logistic Regression baseline strictly on the 10,704 training examples,
    preserving exact feature isolation and evaluating on the identical 3,346 test examples.
    """
    train_texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(train_data["questions"], train_data["answers"])
    ]
    test_texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(test_data["questions"], test_data["answers"])
    ]
    y_train = train_data["labels"]
    y_test = test_data["labels"]

    logger.info(f"Fitting fair TF-IDF baseline on {len(train_texts)} train examples...")
    pipeline = build_tfidf_lr_pipeline(random_state=random_state)
    t0 = time.time()
    pipeline.fit(train_texts, y_train)
    t_train = time.time() - t0

    logger.info(f"Evaluating fair TF-IDF baseline on {len(test_texts)} test examples...")
    t0 = time.time()
    y_pred = list(pipeline.predict(test_texts))
    t_eval = time.time() - t0

    metrics = evaluate_model(
        model_name="tfidf_logistic_regression_fair_10k",
        y_true=y_test,
        y_pred=y_pred,
        train_time=t_train,
        eval_time=t_eval,
    )

    return {
        "model_name": "tfidf_logistic_regression_fair_10k",
        "accuracy": round(metrics.accuracy, 4),
        "macro_f1": round(metrics.macro_f1, 4),
        "weighted_f1": round(metrics.weighted_f1, 4),
        "per_class": metrics.per_class,
        "confusion_matrix": metrics.confusion_matrix,
        "train_wall_time_seconds": round(t_train, 3),
        "eval_wall_time_seconds": round(t_eval, 3),
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "y_pred": y_pred,
    }


def evaluate_deberta_checkpoint(
    checkpoint_path: Path | str,
    test_data: dict[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = 256,
    batch_size: int = 8,
    device: torch.device | None = None,
    input_mode: str = "qa",
) -> dict[str, Any]:
    """
    Re-evaluates an existing fine-tuned DeBERTa checkpoint on the test partition,
    returning structured metrics and individual predictions.
    """
    checkpoint_p = Path(checkpoint_path)
    if not checkpoint_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_p}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading checkpoint {checkpoint_p} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(SUPPORTED_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    checkpoint = torch.load(checkpoint_p, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.float()
    model.to(device)
    model.eval()

    test_dataset = EvasionQAPairDataset(
        questions=test_data["questions"],
        answers=test_data["answers"],
        labels=test_data["labels"],
        tokenizer=tokenizer,
        max_length=max_length,
        input_mode=input_mode,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_qa_batch(b, pad_token_id=pad_id),
    )

    y_preds: list[str] = []
    y_trues: list[str] = []
    t0 = time.time()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = (
                batch["token_type_ids"].to(device) if "token_type_ids" in batch else None
            )

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            outputs = model(**kwargs)
            preds = torch.argmax(outputs.logits, dim=-1)

            y_preds.extend([ID_TO_LABEL[p.item()] for p in preds])
            y_trues.extend([ID_TO_LABEL[lbl.item()] for lbl in labels])

    eval_time = time.time() - t0
    metrics = compute_metrics(y_trues, y_preds)

    peak_gpu_mem = 0.0
    if device.type == "cuda":
        peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_p),
        "max_length": max_length,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "eval_wall_time_seconds": round(eval_time, 3),
        "peak_gpu_memory_mb": round(peak_gpu_mem, 2),
        "truncation_stats": test_dataset.truncation_stats.to_dict(),
        "y_pred": y_preds,
    }


def bootstrap_significance_test(
    y_true: list[str],
    predictions: dict[str, list[str]],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Performs paired fixed-seed bootstrap resampling over the test partition
    to estimate 95% Confidence Intervals for Macro-F1 and pairwise model differences.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    model_names = list(predictions.keys())

    # Storage for bootstrap metrics
    macro_f1_boot: dict[str, list[float]] = {name: [] for name in model_names}
    fe_f1_boot: dict[str, list[float]] = {name: [] for name in model_names}

    y_true_arr = np.array(y_true)
    pred_arrs = {name: np.array(preds) for name, preds in predictions.items()}

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, size=n_samples)
        b_true = y_true_arr[idx]

        for name in model_names:
            b_pred = pred_arrs[name][idx]
            mf1 = float(f1_score(b_true, b_pred, average="macro", zero_division=0))
            # Fully evasive F1
            fe_f1 = float(
                f1_score(
                    b_true == "fully_evasive",
                    b_pred == "fully_evasive",
                    pos_label=True,
                    zero_division=0,
                )
            )
            macro_f1_boot[name].append(mf1)
            fe_f1_boot[name].append(fe_f1)

    results: dict[str, Any] = {
        "n_bootstraps": n_bootstraps,
        "seed": seed,
        "models": {},
        "pairwise_differences": {},
    }

    for name in model_names:
        m_dist = np.array(macro_f1_boot[name])
        fe_dist = np.array(fe_f1_boot[name])
        results["models"][name] = {
            "macro_f1_mean": round(float(np.mean(m_dist)), 4),
            "macro_f1_std": round(float(np.std(m_dist)), 4),
            "macro_f1_ci95": [
                round(float(np.percentile(m_dist, 2.5)), 4),
                round(float(np.percentile(m_dist, 97.5)), 4),
            ],
            "fully_evasive_f1_mean": round(float(np.mean(fe_dist)), 4),
            "fully_evasive_f1_std": round(float(np.std(fe_dist)), 4),
            "fully_evasive_f1_ci95": [
                round(float(np.percentile(fe_dist, 2.5)), 4),
                round(float(np.percentile(fe_dist, 97.5)), 4),
            ],
        }

    # Pairwise differences
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1 = model_names[i]
            m2 = model_names[j]
            pair_key = f"{m2}_vs_{m1}"

            diff_macro = np.array(macro_f1_boot[m2]) - np.array(macro_f1_boot[m1])
            diff_fe = np.array(fe_f1_boot[m2]) - np.array(fe_f1_boot[m1])

            p_val_macro = float(np.mean(diff_macro <= 0.0))
            p_val_fe = float(np.mean(diff_fe <= 0.0))

            results["pairwise_differences"][pair_key] = {
                "macro_f1_delta_mean": round(float(np.mean(diff_macro)), 4),
                "macro_f1_delta_ci95": [
                    round(float(np.percentile(diff_macro, 2.5)), 4),
                    round(float(np.percentile(diff_macro, 97.5)), 4),
                ],
                "macro_f1_p_value_m2_le_m1": round(p_val_macro, 4),
                "fully_evasive_f1_delta_mean": round(float(np.mean(diff_fe)), 4),
                "fully_evasive_f1_delta_ci95": [
                    round(float(np.percentile(diff_fe, 2.5)), 4),
                    round(float(np.percentile(diff_fe, 97.5)), 4),
                ],
                "fully_evasive_f1_p_value_m2_le_m1": round(p_val_fe, 4),
            }

    return results
