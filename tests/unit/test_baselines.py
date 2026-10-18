"""
FinVeritas: Unit Tests for Fast NLP Baselines and Leakage-Aware Splitting.
Tests question normalization, StratifiedGroupKFold isolation,
majority baseline behavior, pipeline training, metrics serialization,
model persistence, and inference contracts.
"""

import json
import tempfile
import unittest
from pathlib import Path

import joblib

from finveritas.nlp_evasion.baselines import (
    SUPPORTED_CLASSES,
    MajorityClassBaseline,
    build_tfidf_lr_pipeline,
    create_leakage_aware_split,
    evaluate_model,
    load_baseline_model,
    normalize_question_for_grouping,
    predict_evasion,
)


class TestNLPBaselines(unittest.TestCase):
    def test_normalize_question_for_grouping(self):
        raw_q = "  What   was \t Q3  REVENUE? \n"
        norm = normalize_question_for_grouping(raw_q)
        self.assertEqual(norm, "what was q3 revenue?")

        # Unicode NFKC normalization
        unicode_q = "What is the 𝖥𝗂𝗇𝖺𝗇𝖼𝗂𝖺𝗅 guidance?"
        norm_unicode = normalize_question_for_grouping(unicode_q)
        self.assertEqual(norm_unicode, "what is the financial guidance?")

    def test_leakage_aware_split_no_overlap(self):
        # 25 samples ensuring >= 5 samples per class for 5-fold stratification
        questions = [
            "What was revenue?",
            "What was revenue?  ",
            "What was revenue?",
            "Why did churn rise?",
            "Why did churn rise?",
            "Will margins expand?",
            "Will margins expand?",
            "Guidance for Q4?",
            "Guidance for Q4?",
            "CAPEX outlook?",
            "CAPEX outlook?",
            "Pricing pressure?",
            "Pricing pressure?",
            "Hiring plans?",
            "Hiring plans?",
            "Tax rate expectation?",
            "Dividend payout?",
            "Supply chain risks?",
            "R&D spend?",
            "Debt maturity profile?",
            "Free cash flow?",
            "Headcount target?",
            "Tariff exposure?",
            "M&A pipeline?",
            "Interest expense?",
        ]
        # 10 direct, 10 intermediate, 5 fully_evasive (>= 5 per class)
        labels = [
            "direct",
            "direct",
            "direct",
            "intermediate",
            "intermediate",
            "intermediate",
            "direct",
            "intermediate",
            "fully_evasive",
            "direct",
            "intermediate",
            "fully_evasive",
            "direct",
            "intermediate",
            "fully_evasive",
            "direct",
            "intermediate",
            "fully_evasive",
            "direct",
            "intermediate",
            "fully_evasive",
            "direct",
            "intermediate",
            "direct",
            "intermediate",
        ]

        train_idx, test_idx, meta = create_leakage_aware_split(
            questions=questions,
            labels=labels,
            n_splits=5,
            random_state=42,
            fold_to_use=0,
        )

        norm_groups = [normalize_question_for_grouping(q) for q in questions]
        train_groups = set(norm_groups[i] for i in train_idx)
        test_groups = set(norm_groups[i] for i in test_idx)

        # Zero normalized-question overlap assertion
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertEqual(meta["question_group_overlap"], 0)
        self.assertEqual(len(train_idx) + len(test_idx), len(questions))

    def test_majority_class_baseline(self):
        clf = MajorityClassBaseline()
        # Unfitted call raises error
        with self.assertRaises(ValueError):
            clf.predict(["test question"])

        X_train = ["q1", "q2", "q3", "q4"]
        y_train = ["direct", "intermediate", "direct", "fully_evasive"]

        clf.fit(X_train, y_train)
        self.assertEqual(clf.majority_label_, "direct")

        preds = clf.predict(["new_q1", "new_q2", "new_q3"])
        self.assertEqual(preds, ["direct", "direct", "direct"])

    def test_tfidf_pipeline_train_and_evaluate(self):
        X_train = [
            "Question: Q1 revenue?\nAnswer: Revenue was $5B.",
            "Question: Guidance?\nAnswer: We do not guide.",
            "Question: Margins?\nAnswer: Margins may improve somewhat.",
            "Question: Churn?\nAnswer: Customer churn was low.",
            "Question: Debt?\nAnswer: We will not discuss our debt today.",
            "Question: Capex?\nAnswer: Capex was $200M.",
        ]
        y_train = [
            "direct",
            "fully_evasive",
            "intermediate",
            "direct",
            "fully_evasive",
            "direct",
        ]

        X_test = [
            "Question: Q1 revenue?\nAnswer: Revenue was strong.",
            "Question: Dividend?\nAnswer: Refuse to comment.",
        ]
        y_test = ["direct", "fully_evasive"]

        pipeline = build_tfidf_lr_pipeline(random_state=42, max_iter=200)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        self.assertEqual(len(preds), 2)

        metrics = evaluate_model(
            model_name="Test Model",
            y_true=y_test,
            y_pred=preds,
            train_time=0.1,
            eval_time=0.01,
        )

        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertEqual(len(metrics.confusion_matrix), 3)

        # JSON serializability check
        json_str = json.dumps(metrics.__dict__)
        self.assertTrue(len(json_str) > 0)

    def test_model_persistence_and_loading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.joblib"
            pipeline = build_tfidf_lr_pipeline(random_state=42, max_iter=200)
            X = [
                "Question: Revenue?\nAnswer: $1B revenue this quarter.",
                "Question: Outlook?\nAnswer: We do not guide on forward periods.",
                "Question: Margins?\nAnswer: Operating margins may expand slightly.",
                "Question: Capex?\nAnswer: Capex was $50M for the year.",
                "Question: M&A?\nAnswer: We never speculate on potential acquisitions.",
                "Question: Cash flow?\nAnswer: Free cash flow was positive.",
                "Question: Pricing?\nAnswer: Some pricing adjustments may occur.",
                "Question: Headcount?\nAnswer: We will not share hiring numbers.",
            ]
            y = [
                "direct",
                "fully_evasive",
                "intermediate",
                "direct",
                "fully_evasive",
                "direct",
                "intermediate",
                "fully_evasive",
            ]
            pipeline.fit(X, y)

            joblib.dump(pipeline, model_path)
            self.assertTrue(model_path.is_file())

            loaded = load_baseline_model(model_path)
            orig_preds = pipeline.predict(["Question: Revenue?\nAnswer: $1B revenue this quarter."])
            load_preds = loaded.predict(["Question: Revenue?\nAnswer: $1B revenue this quarter."])
            self.assertEqual(orig_preds.tolist(), load_preds.tolist())

    def test_load_baseline_model_missing_raises(self):
        nonexistent = Path("artifacts/nonexistent_model.joblib")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_baseline_model(nonexistent)
        self.assertIn("Fitted baseline model artifact not found", str(ctx.exception))

    def test_predict_evasion_contract(self):
        pipeline = build_tfidf_lr_pipeline(random_state=42, max_iter=200)
        X = [
            "Question: Revenue?\nAnswer: $1B revenue this quarter.",
            "Question: Outlook?\nAnswer: We do not guide on forward periods.",
            "Question: Margins?\nAnswer: Operating margins may expand slightly.",
            "Question: Capex?\nAnswer: Capex was $50M for the year.",
            "Question: M&A?\nAnswer: We never speculate on potential acquisitions.",
            "Question: Cash flow?\nAnswer: Free cash flow was positive.",
            "Question: Pricing?\nAnswer: Some pricing adjustments may occur.",
            "Question: Headcount?\nAnswer: We will not share hiring numbers.",
        ]
        y = [
            "direct",
            "fully_evasive",
            "intermediate",
            "direct",
            "fully_evasive",
            "direct",
            "intermediate",
            "fully_evasive",
        ]
        pipeline.fit(X, y)

        res = predict_evasion(pipeline, question="What was Q3 revenue?", answer="Revenue was $5B.")

        self.assertIn("predicted_label", res)
        self.assertIn(res["predicted_label"], SUPPORTED_CLASSES)
        self.assertIn("uncalibrated_probabilities", res)
        probs = res["uncalibrated_probabilities"]
        for cls_name in SUPPORTED_CLASSES:
            self.assertIn(cls_name, probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
        self.assertIn("disclaimer", res)


if __name__ == "__main__":
    unittest.main()
