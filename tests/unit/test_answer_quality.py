"""
FinVeritas: Unit Tests for Answer-Quality NLP Baseline.

Tests label mapping, schema validation, zero-leakage grouped cross-validation,
feature vectorization, multiclass metrics calculation, and paired group bootstrap.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from finveritas.verification import (
    CANONICAL_CLASSES,
    LABEL_MAPPING,
    AnswerQualityRecord,
    build_classifier_pipeline,
    compute_multiclass_metrics,
    load_and_validate_results,
    paired_group_bootstrap,
)


class TestAnswerQuality(unittest.TestCase):
    def setUp(self):
        self.sample_records = [
            AnswerQualityRecord(
                financebench_id="financebench_001",
                model_name="gpt-4",
                eval_mode="direct",
                temp=0.0,
                question="What was Corp A operating margin in 2022?",
                model_answer="Corp A operating margin was 18.2%.",
                raw_label="Correct Answer",
                normalized_label="correct",
            ),
            AnswerQualityRecord(
                financebench_id="financebench_001",
                model_name="claude-2",
                eval_mode="direct",
                temp=0.0,
                question="What was Corp A operating margin in 2022?",
                model_answer="The operating margin was 12.0%.",
                raw_label="Incorrect Answer",
                normalized_label="incorrect",
            ),
            AnswerQualityRecord(
                financebench_id="financebench_002",
                model_name="gpt-4",
                eval_mode="direct",
                temp=0.0,
                question="What was Corp B capital expenditure?",
                model_answer="I do not have sufficient information in the context.",
                raw_label="Refusal",
                normalized_label="refusal",
            ),
            AnswerQualityRecord(
                financebench_id="financebench_002",
                model_name="claude-2",
                eval_mode="direct",
                temp=0.0,
                question="What was Corp B capital expenditure?",
                model_answer="Capital expenditure was $1.5 billion.",
                raw_label="Correct Answer",
                normalized_label="correct",
            ),
        ]

    def test_canonical_classes_and_mapping(self):
        self.assertEqual(CANONICAL_CLASSES, ["correct", "incorrect", "refusal"])
        self.assertEqual(LABEL_MAPPING["Correct Answer"], "correct")
        self.assertEqual(LABEL_MAPPING["Incorrect Answer"], "incorrect")
        self.assertEqual(LABEL_MAPPING["Refusal"], "refusal")

    def test_record_immutability_and_string_coercion(self):
        rec = self.sample_records[0]
        self.assertEqual(rec.financebench_id, "financebench_001")
        self.assertEqual(rec.normalized_label, "correct")
        # Ensure dataclass is frozen
        with self.assertRaises(AttributeError):
            rec.financebench_id = "new_id"  # type: ignore

    def test_load_and_validate_results_with_mock_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            jsonl_file = tmp_path / "model_eval.jsonl"
            rows = [
                {
                    "financebench_id": "fb_101",
                    "model_name": "test_model",
                    "eval_mode": "direct",
                    "temp": 0.0,
                    "question": "What is revenue?",
                    "model_answer": 1577,  # numeric answer to test coercion
                    "label": "Correct Answer",
                },
                {
                    "financebench_id": "fb_102",
                    "model_name": "test_model",
                    "eval_mode": "direct",
                    "temp": 0.0,
                    "question": "What is debt?",
                    "model_answer": "I cannot answer.",
                    "label": "Refusal",
                },
            ]
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            records = load_and_validate_results(tmp_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].model_answer, "1577")
            self.assertEqual(records[0].normalized_label, "correct")
            self.assertEqual(records[1].normalized_label, "refusal")

    def test_load_and_validate_results_catches_invalid_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            jsonl_file = tmp_path / "bad_label.jsonl"
            bad_row = {
                "financebench_id": "fb_201",
                "question": "What is net income?",
                "model_answer": "$500M",
                "label": "Partially Correct",  # invalid label
            }
            with open(jsonl_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(bad_row) + "\n")

            with self.assertRaises(ValueError) as ctx:
                load_and_validate_results(tmp_path)
            self.assertIn("Unknown or missing label", str(ctx.exception))

    def test_zero_question_leakage_in_grouped_kfold(self):
        from sklearn.model_selection import StratifiedGroupKFold

        # Create a synthetic dataset of 25 question groups with 4 completions each (100 rows)
        records = []
        labels_pool = ["correct", "incorrect", "refusal"]
        for q_idx in range(25):
            fb_id = f"q_{q_idx:03d}"
            for m_idx in range(4):
                lbl = labels_pool[(q_idx + m_idx) % 3]
                records.append({
                    "financebench_id": fb_id,
                    "question": f"Question {q_idx}",
                    "answer": f"Answer from model {m_idx}",
                    "label": lbl,
                })

        y = [r["label"] for r in records]
        groups = [r["financebench_id"] for r in records]

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(records, y, groups)):
            train_groups = set(groups[i] for i in train_idx)
            val_groups = set(groups[i] for i in val_idx)
            overlap = train_groups & val_groups
            # Assert zero leakage
            self.assertEqual(len(overlap), 0, f"Fold {fold_idx} has question leakage!")
            self.assertEqual(len(train_groups) + len(val_groups), 25)

    def test_vectorizer_and_classifier_pipeline(self):
        pipe = build_classifier_pipeline(random_state=42)
        X_train = [
            "We do not possess sufficient information to answer.",
            "Operating margin expanded by 3.5% in 2022.",
            "The company reported net loss of $(150) million.",
            "I am unable to locate this figure in the filing.",
            "Total capital expenditures were $1,200M.",
            "The revenue grew by 50% year over year.",
        ]
        y_train = ["refusal", "correct", "incorrect", "refusal", "correct", "incorrect"]

        pipe.fit(X_train, y_train)

        test_samples = [
            "I cannot find this information in the report.",
            "Capital expenditures were $1,200 million.",
        ]
        preds = pipe.predict(test_samples)
        self.assertEqual(len(preds), 2)
        self.assertIn(preds[0], CANONICAL_CLASSES)
        self.assertIn(preds[1], CANONICAL_CLASSES)

    def test_multiclass_metrics_computation(self):
        y_true = ["correct", "correct", "incorrect", "refusal", "refusal"]
        y_pred = ["correct", "incorrect", "incorrect", "refusal", "correct"]

        metrics = compute_multiclass_metrics(y_true, y_pred)

        # 3 out of 5 correct
        self.assertEqual(metrics["accuracy"], 0.6)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertEqual(len(metrics["per_class"]), 3)
        self.assertEqual(metrics["per_class"]["refusal"]["support"], 2)
        self.assertEqual(metrics["per_class"]["correct"]["support"], 2)
        self.assertEqual(metrics["per_class"]["incorrect"]["support"], 1)

        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["classes"], CANONICAL_CLASSES)
        self.assertEqual(len(cm["matrix"]), 3)
        self.assertEqual(len(cm["matrix"][0]), 3)

    def test_paired_group_bootstrap(self):
        # 10 questions with 4 models each (40 rows)
        groups = [f"q_{i // 4}" for i in range(40)]
        y_true = ["correct"] * 20 + ["refusal"] * 20

        # Model A is perfect
        y_pred_a = list(y_true)
        # Model B is wrong half the time
        y_pred_b = ["incorrect"] * 10 + ["correct"] * 10 + ["incorrect"] * 10 + ["refusal"] * 10

        res = paired_group_bootstrap(
            y_true=y_true,
            y_pred_a=y_pred_a,
            y_pred_b=y_pred_b,
            groups=groups,
            n_bootstraps=100,
            random_state=42,
        )

        self.assertIn("mean_delta_accuracy", res)
        self.assertIn("ci_95_accuracy", res)
        self.assertIn("p_value_accuracy", res)
        self.assertGreater(res["mean_delta_accuracy"], 0.0)
        self.assertGreater(res["mean_delta_macro_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
