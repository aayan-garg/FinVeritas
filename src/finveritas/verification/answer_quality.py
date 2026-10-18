"""
FinVeritas: Answer-Quality NLP Baseline Classifier.

Classifies financial question-answering model completions into:
  - 'correct': grounded and verified accurate
  - 'incorrect': inaccurate, hallucinated, or contradicted
  - 'refusal': model explicitly abstained due to lack of evidence

Evaluates using 5-fold group-aware cross-validation grouped by question ID
(financebench_id) to strictly prevent question leakage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

logger = logging.getLogger("answer_quality")

LABEL_MAPPING: dict[str, str] = {
    "Correct Answer": "correct",
    "Incorrect Answer": "incorrect",
    "Refusal": "refusal",
}

CANONICAL_CLASSES: list[str] = ["correct", "incorrect", "refusal"]


@dataclass(frozen=True)
class AnswerQualityRecord:
    """Represents a single model completion record with validated labels."""

    financebench_id: str
    model_name: str
    eval_mode: str
    temp: float
    question: str
    model_answer: str
    raw_label: str
    normalized_label: str


def load_and_validate_results(
    results_dir: Path,
    open_source_path: Path | None = None,
) -> list[AnswerQualityRecord]:
    """
    Loads all results JSONL files, validates schemas and labels, and ensures
    zero target leakage (never including gold answers or evidence strings).
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    # Optionally load canonical question text from financebench_open_source.jsonl
    open_source_questions: dict[str, str] = {}
    if open_source_path and open_source_path.is_file():
        with open(open_source_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    item = json.loads(line_str)
                    fb_id = item.get("financebench_id")
                    if fb_id and "question" in item:
                        open_source_questions[fb_id] = item["question"]

    result_files = sorted(results_dir.glob("*.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No .jsonl files found in {results_dir}")

    records: list[AnswerQualityRecord] = []
    seen_keys: set[tuple[str, str]] = set()

    for file_path in result_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)

                fb_id = data.get("financebench_id") or data.get("id")
                if not fb_id:
                    raise ValueError(
                        f"Missing question ID in {file_path.name} line {line_idx}"
                    )

                raw_label = data.get("label")
                if raw_label not in LABEL_MAPPING:
                    raise ValueError(
                        f"Unknown or missing label '{raw_label}' in "
                        f"{file_path.name} line {line_idx}"
                    )
                norm_label = LABEL_MAPPING[raw_label]

                # Question text resolution
                q_text = data.get("question")
                if not q_text and fb_id in open_source_questions:
                    q_text = open_source_questions[fb_id]
                if not q_text:
                    raise ValueError(
                        f"Missing question text for {fb_id} in {file_path.name} line {line_idx}"
                    )

                # Ensure model answer is converted to string (handles numeric answers like 1577)
                raw_ans = data.get("model_answer", "")
                ans_text = str(raw_ans) if raw_ans is not None else ""

                model_name = str(data.get("model_name", file_path.stem))
                eval_mode = str(data.get("eval_mode", "direct"))
                temp_val = float(data.get("temp", 0.0))

                # Check duplicate records within the same configuration file
                config_key = (file_path.stem, fb_id)
                if config_key in seen_keys:
                    logger.warning(
                        f"Duplicate evaluation record encountered for {config_key}"
                    )
                seen_keys.add(config_key)

                records.append(
                    AnswerQualityRecord(
                        financebench_id=fb_id,
                        model_name=model_name,
                        eval_mode=eval_mode,
                        temp=temp_val,
                        question=str(q_text).strip(),
                        model_answer=ans_text.strip(),
                        raw_label=str(raw_label).strip(),
                        normalized_label=norm_label,
                    )
                )

    return records


def build_text_vectorizer() -> FeatureUnion:
    """
    Builds a lightweight, informative text feature extractor combining
    word n-grams and sub-word character n-grams.
    """
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_classifier_pipeline(
    feature_extractor: FeatureUnion | None = None,
    class_weight: str | None = "balanced",
    random_state: int = 42,
    c_reg: float = 1.0,
) -> Pipeline:
    """Creates a scikit-learn pipeline for text classification."""
    fe = feature_extractor or build_text_vectorizer()
    clf = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
        C=c_reg,
    )
    return Pipeline([("vectorizer", fe), ("classifier", clf)])


def compute_multiclass_metrics(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str] = CANONICAL_CLASSES,
) -> dict[str, Any]:
    """Computes comprehensive classification metrics across all classes."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, labels=classes, average="weighted", zero_division=0
    )

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class = {}
    for cls_name, p, r, f, s in zip(classes, p_per, r_per, f1_per, sup_per):
        per_class[cls_name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm.tolist(),
        },
    }


def paired_group_bootstrap(
    y_true: list[str],
    y_pred_a: list[str],
    y_pred_b: list[str],
    groups: list[str],
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Performs a paired cluster bootstrap over question groups to estimate
    statistical uncertainty in accuracy and macro-F1 differences between two models.
    """
    rng = np.random.RandomState(random_state)
    unique_groups = np.array(sorted(list(set(groups))))
    n_groups = len(unique_groups)

    group_to_indices: dict[str, list[int]] = {}
    for idx, grp in enumerate(groups):
        if grp not in group_to_indices:
            group_to_indices[grp] = []
        group_to_indices[grp].append(idx)

    y_true_arr = np.array(y_true)
    y_pred_a_arr = np.array(y_pred_a)
    y_pred_b_arr = np.array(y_pred_b)

    delta_accs: list[float] = []
    delta_macro_f1s: list[float] = []

    for _ in range(n_bootstraps):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sampled_indices: list[int] = []
        for grp in sampled_groups:
            sampled_indices.extend(group_to_indices[grp])

        sub_true = y_true_arr[sampled_indices]
        sub_a = y_pred_a_arr[sampled_indices]
        sub_b = y_pred_b_arr[sampled_indices]

        acc_a = accuracy_score(sub_true, sub_a)
        acc_b = accuracy_score(sub_true, sub_b)
        delta_accs.append(float(acc_a - acc_b))

        f1_a = f1_score(
            sub_true, sub_a, labels=CANONICAL_CLASSES, average="macro", zero_division=0
        )
        f1_b = f1_score(
            sub_true, sub_b, labels=CANONICAL_CLASSES, average="macro", zero_division=0
        )
        delta_macro_f1s.append(float(f1_a - f1_b))

    acc_diffs = np.array(delta_accs)
    f1_diffs = np.array(delta_macro_f1s)

    # 95% bootstrap confidence interval
    ci_acc = (
        round(float(np.percentile(acc_diffs, 2.5)), 4),
        round(float(np.percentile(acc_diffs, 97.5)), 4),
    )
    ci_f1 = (
        round(float(np.percentile(f1_diffs, 2.5)), 4),
        round(float(np.percentile(f1_diffs, 97.5)), 4),
    )

    # Two-sided empirical p-value for hypothesis that difference is 0
    p_acc = float(
        np.mean(acc_diffs <= 0) if np.mean(acc_diffs) > 0 else np.mean(acc_diffs >= 0)
    ) * 2
    p_f1 = float(
        np.mean(f1_diffs <= 0) if np.mean(f1_diffs) > 0 else np.mean(f1_diffs >= 0)
    ) * 2

    return {
        "n_bootstraps": n_bootstraps,
        "n_groups_sampled": n_groups,
        "mean_delta_accuracy": round(float(np.mean(acc_diffs)), 4),
        "ci_95_accuracy": list(ci_acc),
        "p_value_accuracy": round(min(1.0, p_acc), 4),
        "mean_delta_macro_f1": round(float(np.mean(f1_diffs)), 4),
        "ci_95_macro_f1": list(ci_f1),
        "p_value_macro_f1": round(min(1.0, p_f1), 4),
    }


def evaluate_grouped_baselines(
    records: list[AnswerQualityRecord],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Executes 5-fold group-aware cross-validation across all 4 benchmark configurations:
      A. Majority-Class
      B. Question-Only TF-IDF + Logistic Regression
      C. Answer-Only TF-IDF + Logistic Regression
      D. Question + Answer TF-IDF + Logistic Regression
    """
    y_true = [r.normalized_label for r in records]
    groups = [r.financebench_id for r in records]

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    configs: dict[str, Callable[[AnswerQualityRecord], str]] = {
        "majority": lambda r: "",
        "question_only": lambda r: str(r.question),
        "answer_only": lambda r: str(r.model_answer),
        "question_plus_answer": lambda r: f"Question: {r.question}\nAnswer: {r.model_answer}",
    }

    results: dict[str, Any] = {
        "dataset_summary": {
            "total_records": len(records),
            "unique_question_groups": len(set(groups)),
            "label_distribution": {
                lbl: y_true.count(lbl) for lbl in CANONICAL_CLASSES
            },
            "n_splits": n_splits,
            "random_state": random_state,
        },
        "models": {},
    }

    # Store out-of-fold predictions for paired bootstrap comparisons
    oof_predictions_by_model: dict[str, list[str]] = {}

    for config_name, text_fn in configs.items():
        logger.info(f"Training and evaluating configuration '{config_name}'...")
        t_start = time.perf_counter()

        oof_preds: list[str | None] = [None] * len(records)
        fold_evaluations: list[dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(records, y_true, groups)):
            # Assert zero question overlap
            train_groups = set(groups[i] for i in train_idx)
            val_groups = set(groups[i] for i in val_idx)
            group_overlap = train_groups & val_groups
            if group_overlap:
                raise RuntimeError(
                    f"Question leakage detected in fold {fold_idx}: "
                    f"{len(group_overlap)} overlapping groups!"
                )

            y_val_fold = [y_true[i] for i in val_idx]

            if config_name == "majority":
                # Find most frequent training label
                train_labels = [y_true[i] for i in train_idx]
                maj_label = max(set(train_labels), key=train_labels.count)
                fold_preds = [maj_label] * len(val_idx)
            else:
                X_train = [text_fn(records[i]) for i in train_idx]
                y_train = [y_true[i] for i in train_idx]
                X_val = [text_fn(records[i]) for i in val_idx]

                pipe = build_classifier_pipeline(random_state=random_state)
                pipe.fit(X_train, y_train)
                fold_preds = list(pipe.predict(X_val))

            for idx, p in zip(val_idx, fold_preds):
                oof_preds[idx] = p

            fold_metric = compute_multiclass_metrics(y_val_fold, fold_preds)
            fold_evaluations.append({
                "fold_index": fold_idx,
                "train_records": len(train_idx),
                "val_records": len(val_idx),
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "metrics": fold_metric,
            })

        duration = time.perf_counter() - t_start
        assert all(p is not None for p in oof_preds)
        valid_oof_preds = [str(p) for p in oof_preds]
        oof_predictions_by_model[config_name] = valid_oof_preds

        overall_metrics = compute_multiclass_metrics(y_true, valid_oof_preds)

        # Diagnostic model configuration breakdown
        subgroup_diag: dict[str, dict[str, Any]] = {}
        model_names = sorted(list(set(r.model_name for r in records)))
        for m_name in model_names:
            sub_indices = [i for i, r in enumerate(records) if r.model_name == m_name]
            sub_true = [y_true[i] for i in sub_indices]
            sub_preds = [valid_oof_preds[i] for i in sub_indices]
            subgroup_diag[m_name] = {
                "support": len(sub_indices),
                "accuracy": round(float(accuracy_score(sub_true, sub_preds)), 4),
                "macro_f1": round(
                    float(
                        f1_score(
                            sub_true,
                            sub_preds,
                            labels=CANONICAL_CLASSES,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
            }

        results["models"][config_name] = {
            "runtime_seconds": round(duration, 3),
            "overall_metrics": overall_metrics,
            "fold_breakdown": fold_evaluations,
            "subgroup_diagnostic": subgroup_diag,
        }

    # Paired cluster bootstrap comparisons against answer_only (the best model)
    best_model_name = "answer_only"
    bootstrap_results = {}
    for comp_name in ["majority", "question_only", "question_plus_answer"]:
        comp_res = paired_group_bootstrap(
            y_true=y_true,
            y_pred_a=oof_predictions_by_model[best_model_name],
            y_pred_b=oof_predictions_by_model[comp_name],
            groups=groups,
            n_bootstraps=1000,
            random_state=random_state,
        )
        bootstrap_results[f"{best_model_name}_vs_{comp_name}"] = comp_res

    results["paired_group_bootstraps"] = bootstrap_results
    return results
