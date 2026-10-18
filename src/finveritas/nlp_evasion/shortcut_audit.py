"""
FinVeritas: NLP Shortcut Learning and Robustness Audit Module.

Provides utilities for:
1. Strict normalized-answer-disjoint train/validation/test partitioning.
2. Answer template frequency and pseudo-label distribution diagnostics.
3. Performance auditing across overlapping vs unseen answer cohorts.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from finveritas.nlp_evasion.baselines import SUPPORTED_CLASSES
from finveritas.nlp_evasion.fair_benchmark import normalize_answer_text
from finveritas.nlp_evasion.transformer_classifier import compute_metrics

logger = logging.getLogger(__name__)


def create_answer_disjoint_splits(
    parquet_path: Path | str,
    random_state: int = 42,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    """
    Creates a strict train/validation/test partition where identical normalized
    answers CANNOT cross partitions.

    Uses StratifiedGroupKFold with groups defined strictly by normalized answer text.
    Class weights are calculated strictly on the training partition.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet dataset not found at {path}")

    table = pq.read_table(
        path,
        columns=["question", "answer", "eva4b_label"],
    )
    df_raw = table.to_pydict()

    questions = [str(q) for q in df_raw["question"]]
    answers = [str(a) for a in df_raw["answer"]]
    labels = [str(lbl) for lbl in df_raw["eva4b_label"]]

    for lbl in labels:
        if lbl not in SUPPORTED_CLASSES:
            raise ValueError(f"Unknown label '{lbl}' in dataset. Expected {SUPPORTED_CLASSES}")

    # Grouping key: normalized answer text
    norm_answers = [normalize_answer_text(a) for a in answers]

    # Step 1: Base test partition (StratifiedGroupKFold fold 0, 5 splits = 20% test)
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(sgkf_test.split(answers, labels, groups=norm_answers))
    train_val_idx, test_idx = splits[0]

    # Step 2: Split train_val into train and validation grouped by normalized answer
    train_val_answers = [answers[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]
    train_val_groups = [norm_answers[i] for i in train_val_idx]

    n_val_splits = int(round(1.0 / val_fraction))
    sgkf_val = StratifiedGroupKFold(
        n_splits=n_val_splits, shuffle=True, random_state=random_state
    )
    val_splits = list(
        sgkf_val.split(train_val_answers, train_val_labels, groups=train_val_groups)
    )

    inner_train_idx, inner_val_idx = val_splits[0]
    final_train_idx = train_val_idx[inner_train_idx]
    final_val_idx = train_val_idx[inner_val_idx]

    # Step 3: Zero-leakage verification across all 3 sets for answer groups
    train_groups = set(norm_answers[i] for i in final_train_idx)
    val_groups = set(norm_answers[i] for i in final_val_idx)
    test_groups = set(norm_answers[i] for i in test_idx)

    overlap_train_val = train_groups & val_groups
    overlap_train_test = train_groups & test_groups
    overlap_val_test = val_groups & test_groups

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Answer leakage detected! "
            f"Train/Val answer overlap: {len(overlap_train_val)}, "
            f"Train/Test answer overlap: {len(overlap_train_test)}, "
            f"Val/Test answer overlap: {len(overlap_val_test)}"
        )

    # Step 4: Class weights calculated strictly on the training partition
    train_labels = [labels[i] for i in final_train_idx]
    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(SUPPORTED_CLASSES)),
        y=np.array(train_labels),
    )
    class_weights_dict = {
        cls_name: float(w) for cls_name, w in zip(SUPPORTED_CLASSES, computed_weights)
    }

    return {
        "train": {
            "questions": [questions[i] for i in final_train_idx],
            "answers": [answers[i] for i in final_train_idx],
            "labels": [labels[i] for i in final_train_idx],
            "indices": final_train_idx.tolist(),
        },
        "val": {
            "questions": [questions[i] for i in final_val_idx],
            "answers": [answers[i] for i in final_val_idx],
            "labels": [labels[i] for i in final_val_idx],
            "indices": final_val_idx.tolist(),
        },
        "test": {
            "questions": [questions[i] for i in test_idx],
            "answers": [answers[i] for i in test_idx],
            "labels": [labels[i] for i in test_idx],
            "indices": test_idx.tolist(),
        },
        "class_weights": class_weights_dict,
        "metadata": {
            "partitioning_type": "normalized_answer_disjoint",
            "total_samples": len(questions),
            "train_samples": len(final_train_idx),
            "val_samples": len(final_val_idx),
            "test_samples": len(test_idx),
            "unique_answer_groups_train": len(train_groups),
            "unique_answer_groups_val": len(val_groups),
            "unique_answer_groups_test": len(test_groups),
            "answer_group_overlap_train_val": 0,
            "answer_group_overlap_train_test": 0,
            "answer_group_overlap_val_test": 0,
            "random_state": random_state,
        },
    }


def audit_answer_templates(
    answers: list[str],
    labels: list[str],
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """
    Identifies the most frequent normalized answer templates across the dataset,
    their frequencies, and their pseudo-label distributions.
    Protects privacy by summarizing only formulaic templates without disclosing private raw text.
    """
    norm_answers = [normalize_answer_text(a) for a in answers]
    counts = Counter(norm_answers)

    answer_to_labels: dict[str, list[str]] = {}
    for a, lbl in zip(norm_answers, labels):
        answer_to_labels.setdefault(a, []).append(lbl)

    templates: list[dict[str, Any]] = []
    for ans_text, total_count in counts.most_common(top_n):
        if total_count <= 1:
            continue
        lbl_counts = dict(Counter(answer_to_labels[ans_text]))
        dominant_lbl = max(lbl_counts.items(), key=lambda x: x[1])[0]
        dominant_pct = round(lbl_counts[dominant_lbl] / total_count, 4)

        # Truncate template display string for safe presentation
        display_text = ans_text[:80] + ("..." if len(ans_text) > 80 else "")

        templates.append(
            {
                "template": display_text,
                "total_count": total_count,
                "label_distribution": lbl_counts,
                "dominant_label": dominant_lbl,
                "dominant_label_purity": dominant_pct,
            }
        )

    return templates


def evaluate_answer_template_subsets(
    y_true: list[str],
    y_pred: list[str],
    is_overlapping: list[bool],
) -> dict[str, Any]:
    """
    Evaluates classification performance separately on:
    1. Test samples whose normalized answer appeared in training (overlapping answer template).
    2. Test samples whose normalized answer was completely unseen during training.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(is_overlapping):
        raise ValueError("Length mismatch between y_true, y_pred, and is_overlapping")

    idx_overlap = [i for i, flag in enumerate(is_overlapping) if flag]
    idx_unseen = [i for i, flag in enumerate(is_overlapping) if not flag]

    def subset_metrics(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"sample_count": 0, "accuracy": 0.0, "macro_f1": 0.0}
        sub_true = [y_true[i] for i in indices]
        sub_pred = [y_pred[i] for i in indices]
        m = compute_metrics(sub_true, sub_pred)
        m["sample_count"] = len(indices)
        return m

    return {
        "overlapping_answers": subset_metrics(idx_overlap),
        "unseen_answers": subset_metrics(idx_unseen),
        "overall": compute_metrics(y_true, y_pred),
    }
