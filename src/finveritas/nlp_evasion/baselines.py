"""
FinVeritas: Fast, Reproducible NLP Baselines for Earnings-Call Question Evasion.

Implements leakage-aware question-grouped splitting (StratifiedGroupKFold)
and two CPU-only baselines:
1. Majority-Class Minimal Baseline
2. Word + Char TF-IDF FeatureUnion with Balanced Logistic Regression

All evaluated strictly on held-out test rows against Eva-4B-V2 pseudo-labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

logger = logging.getLogger(__name__)

SUPPORTED_CLASSES = ("direct", "intermediate", "fully_evasive")


def normalize_question_for_grouping(text: str) -> str:
    """
    Normalizes question text for group leakage prevention:
    Unicode NFKC normalization, lowercase, whitespace collapsing and stripping.
    Does NOT use target labels.
    """
    norm = unicodedata.normalize("NFKC", str(text))
    return " ".join(norm.lower().split())


def create_leakage_aware_split(
    questions: list[str],
    labels: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    fold_to_use: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Creates a StratifiedGroupKFold split grouping identical normalized questions
    together to prevent question-level leakage between train and test sets.
    """
    groups = [normalize_question_for_grouping(q) for q in questions]
    unique_groups = len(set(groups))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(questions, labels, groups=groups))

    if fold_to_use >= len(splits):
        raise ValueError(f"fold_to_use={fold_to_use} out of range for n_splits={n_splits}")

    train_idx, test_idx = splits[fold_to_use]

    # Verification: Assert zero group overlap
    train_groups = set(groups[i] for i in train_idx)
    test_groups = set(groups[i] for i in test_idx)
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError(
            f"Data leakage detected! {len(overlap)} normalized question groups overlap "
            "between train and test sets."
        )

    split_metadata = {
        "split_method": "StratifiedGroupKFold",
        "n_splits": n_splits,
        "test_fold_index": fold_to_use,
        "random_state": random_state,
        "total_samples": len(labels),
        "unique_question_groups": unique_groups,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "test_fraction": round(len(test_idx) / len(labels), 4),
        "train_class_counts": dict(Counter([labels[i] for i in train_idx])),
        "test_class_counts": dict(Counter([labels[i] for i in test_idx])),
        "question_group_overlap": 0,
    }

    return train_idx, test_idx, split_metadata


class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """
    Predicts the most frequent training label for every test example.
    """

    def __init__(self):
        self.majority_label_: str | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: list[str], y: list[str]) -> MajorityClassBaseline:
        counter = Counter(y)
        self.majority_label_ = counter.most_common(1)[0][0]
        self.classes_ = np.array(sorted(counter.keys()))
        return self

    def predict(self, X: list[str]) -> list[str]:
        if self.majority_label_ is None:
            raise ValueError("MajorityClassBaseline is not fitted yet.")
        return [self.majority_label_] * len(X)


def build_tfidf_lr_pipeline(
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Constructs Word + Char TF-IDF FeatureUnion with Balanced Logistic Regression.
    - Word TF-IDF: unigram and bigram features (1, 2)
    - Char TF-IDF: character n-grams of length 3-5 (char_wb)
    - Balanced class weighting for minority handling
    """
    feature_union = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=40000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    ngram_range=(3, 5),
                    analyzer="char_wb",
                    min_df=5,
                    max_features=60000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    return Pipeline([("features", feature_union), ("classifier", clf)])


@dataclass
class ModelMetrics:
    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    class_names: list[str]
    train_wall_time_seconds: float
    eval_wall_time_seconds: float


def evaluate_model(
    model_name: str,
    y_true: list[str],
    y_pred: list[str],
    train_time: float,
    eval_time: float,
    class_names: list[str] = list(SUPPORTED_CLASSES),
) -> ModelMetrics:
    """
    Computes accuracy, macro-F1, weighted-F1, per-class metrics, and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average=None, zero_division=0
    )

    per_class_dict: dict[str, dict[str, float]] = {}
    for i, c in enumerate(class_names):
        per_class_dict[c] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    # Averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average="macro", zero_division=0
    )
    weight_p, weight_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=class_names).tolist()

    return ModelMetrics(
        model_name=model_name,
        accuracy=round(acc, 4),
        macro_f1=round(float(macro_f1), 4),
        weighted_f1=round(float(weighted_f1), 4),
        per_class=per_class_dict,
        confusion_matrix=cm,
        class_names=class_names,
        train_wall_time_seconds=round(train_time, 3),
        eval_wall_time_seconds=round(eval_time, 3),
    )


def run_baseline_experiments(
    parquet_path: Path,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Executes Experiment A (Majority Class) and Experiment B (TF-IDF + Logistic Regression)
    end-to-end on held-out test data.
    """
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet dataset not found at: {parquet_path}")

    logger.info(f"Loading dataset columns [question, answer, eva4b_label] from {parquet_path}...")
    tbl = pq.read_table(parquet_path, columns=["question", "answer", "eva4b_label"])
    questions = tbl["question"].to_pylist()
    answers = tbl["answer"].to_pylist()
    labels = tbl["eva4b_label"].to_pylist()

    # 1. Leakage-aware split
    train_idx, test_idx, split_meta = create_leakage_aware_split(
        questions=questions,
        labels=labels,
        n_splits=5,
        random_state=42,
        fold_to_use=0,
    )
    logger.info(
        f"Split complete: Train={split_meta['train_samples']}, Test={split_meta['test_samples']} "
        f"({split_meta['test_fraction']*100:.1f}%) | Groups={split_meta['unique_question_groups']}"
    )

    # 2. Prepare text features strictly from question and answer
    train_texts = [f"Question: {questions[i]}\nAnswer: {answers[i]}" for i in train_idx]
    test_texts = [f"Question: {questions[i]}\nAnswer: {answers[i]}" for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    # 3. Experiment A: Majority Class
    logger.info("Training Experiment A: Majority-Class Baseline...")
    majority_model = MajorityClassBaseline()
    t0 = time.time()
    majority_model.fit(train_texts, y_train)
    t_maj_train = time.time() - t0

    t0 = time.time()
    maj_preds = majority_model.predict(test_texts)
    t_maj_eval = time.time() - t0

    maj_metrics = evaluate_model(
        model_name="Majority Class Baseline",
        y_true=y_test,
        y_pred=maj_preds,
        train_time=t_maj_train,
        eval_time=t_maj_eval,
    )

    # 4. Experiment B: TF-IDF + Logistic Regression
    logger.info("Training Experiment B: Word+Char TF-IDF + Balanced Logistic Regression...")
    tfidf_pipeline = build_tfidf_lr_pipeline(random_state=42, max_iter=1000)
    t0 = time.time()
    tfidf_pipeline.fit(train_texts, y_train)
    t_tfidf_train = time.time() - t0

    t0 = time.time()
    tfidf_preds = tfidf_pipeline.predict(test_texts)
    t_tfidf_eval = time.time() - t0

    tfidf_metrics = evaluate_model(
        model_name="TF-IDF + Logistic Regression",
        y_true=y_test,
        y_pred=tfidf_preds,
        train_time=t_tfidf_train,
        eval_time=t_tfidf_eval,
    )

    # Environment and configuration metadata
    config_record = {
        "dataset_path": str(parquet_path),
        "target_field": "eva4b_label",
        "input_fields": ["question", "answer"],
        "prohibited_fields_excluded": ["choices", "eva4b_label_letter", "uid"],
        "label_nature": (
            "Eva-4B-V2 model-generated pseudo-labels "
            "(weak labels, not human ground truth)"
        ),
        "split_configuration": split_meta,
        "models": {
            "majority_class": {
                "class_name": "MajorityClassBaseline",
                "predicted_label": majority_model.majority_label_,
            },
            "tfidf_logistic_regression": {
                "word_tfidf": {
                    "ngram_range": [1, 2],
                    "min_df": 3,
                    "max_features": 40000,
                    "sublinear_tf": True,
                },
                "char_tfidf": {
                    "ngram_range": [3, 5],
                    "analyzer": "char_wb",
                    "min_df": 5,
                    "max_features": 60000,
                    "sublinear_tf": True,
                },
                "classifier": {
                    "class": "LogisticRegression",
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": 42,
                },
            },
        },
        "environment": {
            "python_version": sys.version,
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cpu_only": True,
            "gpu_used": False,
        },
    }

    results = {
        "configuration": config_record,
        "metrics": {
            "majority_class": asdict(maj_metrics),
            "tfidf_logistic_regression": asdict(tfidf_metrics),
        },
    }

    # Save artifacts if directory provided
    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_path = artifacts_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_record, f, indent=2)

        metrics_path = artifacts_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results["metrics"], f, indent=2)

        cm_path = artifacts_dir / "confusion_matrix.json"
        cm_data = {
            "class_names": list(SUPPORTED_CLASSES),
            "majority_class": maj_metrics.confusion_matrix,
            "tfidf_logistic_regression": tfidf_metrics.confusion_matrix,
        }
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(cm_data, f, indent=2)

        logger.info(
            f"Artifacts saved to {artifacts_dir}: "
            "config.json, metrics.json, confusion_matrix.json"
        )

    return results


def train_and_save_baseline_model(
    parquet_path: Path,
    model_save_path: Path,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Fits the TF-IDF + Balanced Logistic Regression pipeline strictly on the
    80% train split and saves the fitted model to model_save_path using joblib.
    """
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet dataset not found at: {parquet_path}")

    logger.info(f"Loading dataset from {parquet_path}...")
    tbl = pq.read_table(parquet_path, columns=["question", "answer", "eva4b_label"])
    questions = tbl["question"].to_pylist()
    answers = tbl["answer"].to_pylist()
    labels = tbl["eva4b_label"].to_pylist()

    train_idx, _, _ = create_leakage_aware_split(
        questions=questions,
        labels=labels,
        n_splits=5,
        random_state=random_state,
        fold_to_use=0,
    )

    train_texts = [f"Question: {questions[i]}\nAnswer: {answers[i]}" for i in train_idx]
    y_train = [labels[i] for i in train_idx]

    logger.info(f"Fitting pipeline on {len(train_texts)} training examples...")
    pipeline = build_tfidf_lr_pipeline(random_state=random_state, max_iter=max_iter)
    pipeline.fit(train_texts, y_train)

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    logger.info(f"Saved fitted model to {model_save_path}")
    return pipeline


def load_baseline_model(model_path: Path) -> Pipeline:
    """
    Loads a fitted baseline pipeline from a joblib artifact file.
    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Fitted baseline model artifact not found at: {model_path}\n"
            "Please train the model first by running:\n"
            "  python scripts/predict_evasion.py --train"
        )
    return joblib.load(model_path)


def predict_evasion(
    pipeline: Pipeline,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """
    Predicts question evasion using a fitted baseline pipeline.
    Returns predicted label and uncalibrated class probabilities.
    """
    formatted_text = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
    pred_label = str(pipeline.predict([formatted_text])[0])

    probs = pipeline.predict_proba([formatted_text])[0]
    classes = list(pipeline.classes_)

    prob_dict = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(classes, probs)
    }

    return {
        "question": question.strip(),
        "answer": answer.strip(),
        "predicted_label": pred_label,
        "uncalibrated_probabilities": prob_dict,
        "disclaimer": (
            "These scores represent linear model decision scores trained on "
            "Eva-4B-V2 pseudo-labels. They are uncalibrated and must not be "
            "interpreted as verified confidence estimates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinVeritas NLP Evasion Baselines (CPU-Only, Leakage-Aware)"
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet"),
        help="Path to the EvasionBench Parquet file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/initial_baseline"),
        help="Directory to save output JSON metrics and configs.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        results = run_baseline_experiments(args.parquet_path, artifacts_dir=args.artifacts_dir)
    except Exception as e:
        logger.error(f"Baseline experiment failed: {e}")
        return 1

    # Print summary
    cfg = results["configuration"]
    split = cfg["split_configuration"]
    m_maj = results["metrics"]["majority_class"]
    m_tfidf = results["metrics"]["tfidf_logistic_regression"]

    print("\n" + "=" * 65)
    print("FINVERITAS NLP EVASION BASELINE EXPERIMENT RESULTS (CPU-ONLY)")
    print("=" * 65)
    print(f"Dataset:       {cfg['dataset_path']} ({split['total_samples']:,} samples)")
    print(
        f"Split:         Train={split['train_samples']:,} (80%), "
        f"Test={split['test_samples']:,} (20%)"
    )
    print(
        f"Groups:        {split['unique_question_groups']:,} "
        "unique normalized questions (0 leakage)"
    )
    print(f"Label Type:    {cfg['label_nature']}")
    py_ver = sys.version.split()[0]
    sklearn_ver = cfg['environment']['scikit_learn_version']
    print(f"Environment:   scikit-learn {sklearn_ver} on Python {py_ver}")

    print("\nTest Class Distribution:")
    for c, cnt in split["test_class_counts"].items():
        print(f"  - {c:14s}: {cnt:5,d} ({cnt/split['test_samples']*100:5.2f}%)")

    print("\n" + "-" * 65)
    print("EXPERIMENT A: MAJORITY CLASS BASELINE ('direct')")
    print("-" * 65)
    print(f"Accuracy:      {m_maj['accuracy']*100:.2f}%")
    print(f"Macro-F1:      {m_maj['macro_f1']:.4f}")
    print(f"Weighted-F1:   {m_maj['weighted_f1']:.4f}")
    t_m_tr = m_maj['train_wall_time_seconds']
    t_m_ev = m_maj['eval_wall_time_seconds']
    print(f"Wall Time:     Train={t_m_tr:.3f}s, Eval={t_m_ev:.3f}s")
    print("Per-Class Scores:")
    for c, sc in m_maj["per_class"].items():
        p, r, f = sc["precision"], sc["recall"], sc["f1"]
        supp = sc["support"]
        print(f"  - {c:14s} P={p:.4f}  R={r:.4f}  F1={f:.4f}  (N={supp})")

    print("\n" + "-" * 65)
    print("EXPERIMENT B: TF-IDF (WORD + CHAR) + BALANCED LOGISTIC REGRESSION")
    print("-" * 65)
    print(f"Accuracy:      {m_tfidf['accuracy']*100:.2f}%")
    print(f"Macro-F1:      {m_tfidf['macro_f1']:.4f}")
    print(f"Weighted-F1:   {m_tfidf['weighted_f1']:.4f}")
    t_tf_tr = m_tfidf['train_wall_time_seconds']
    t_tf_ev = m_tfidf['eval_wall_time_seconds']
    print(f"Wall Time:     Train={t_tf_tr:.2f}s, Eval={t_tf_ev:.2f}s")
    print("Per-Class Scores:")
    for c, sc in m_tfidf["per_class"].items():
        p, r, f = sc["precision"], sc["recall"], sc["f1"]
        supp = sc["support"]
        print(f"  - {c:14s} P={p:.4f}  R={r:.4f}  F1={f:.4f}  (N={supp})")

    print("\nConfusion Matrix (TF-IDF + Logistic Regression):")
    print("                Pred: direct  intermediate  fully_evasive")
    for i, c in enumerate(m_tfidf["class_names"]):
        row = m_tfidf["confusion_matrix"][i]
        print(f"  True {c:13s}   {row[0]:6d}        {row[1]:6d}         {row[2]:6d}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
