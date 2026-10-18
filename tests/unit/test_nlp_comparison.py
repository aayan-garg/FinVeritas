"""
FinVeritas: Unit Tests for Checkpoint 05A Fair Benchmark and Audits.
Tests shared splits, truncation mechanics, normalized answer leakage,
bootstrap significance testing, and metrics serialization.
"""

import unittest
from pathlib import Path

from transformers import AutoTokenizer

from finveritas.nlp_evasion.fair_benchmark import (
    audit_normalized_answer_leakage,
    audit_truncation_mechanics,
    bootstrap_significance_test,
    normalize_answer_text,
    train_fair_tfidf_baseline,
)
from finveritas.nlp_evasion.transformer_classifier import (
    DEFAULT_MODEL_NAME,
    load_and_split_evasion_data,
)


class TestNLPComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    def test_normalize_answer_text(self):
        raw = "  Yes,   WE  are pleased   with Q3. \n"
        norm = normalize_answer_text(raw)
        self.assertEqual(norm, "yes, we are pleased with q3.")

    def test_truncation_mechanics_audit(self):
        # Construct synthetic pairs: one short question + long answer,
        # one long question + short answer
        questions = [
            "Short question?",
            "Long question " + ("word " * 60) + "?",
        ]
        answers = [
            "Long answer " + ("detail " * 60) + ".",
            "Short answer.",
        ]

        summary = audit_truncation_mechanics(
            questions=questions,
            answers=answers,
            tokenizer=self.tokenizer,
            max_length=30,
        )

        self.assertEqual(summary.total_samples, 2)
        self.assertGreater(summary.pair_truncated_count, 0)
        # Verify that question and answer truncation are tracked separately
        self.assertIn("min", summary.question_length_percentiles)
        self.assertIn("max", summary.answer_length_percentiles)
        self.assertIn("p50_median", summary.pair_length_percentiles)
        d = summary.to_dict()
        self.assertIn("pair_truncation_rate", d)
        self.assertIn("answer_truncation_rate", d)
        self.assertIn("question_truncation_rate", d)

    def test_normalized_answer_leakage_audit(self):
        train_ans = ["Yes, we beat revenue guidance.", "No comment on margins."]
        val_ans = ["Different answer here.", "No comment on margins."]
        test_ans = ["Yes, we beat revenue guidance.", "Brand new test answer."]

        train_lbls = ["direct", "fully_evasive"]
        val_lbls = ["direct", "fully_evasive"]
        test_lbls = ["direct", "intermediate"]

        report = audit_normalized_answer_leakage(
            train_answers=train_ans,
            val_answers=val_ans,
            test_answers=test_ans,
            train_labels=train_lbls,
            val_labels=val_lbls,
            test_labels=test_lbls,
        )

        self.assertEqual(report.overlap_train_test, 1)
        self.assertEqual(report.train_test_overlap_samples_in_test, 1)
        # "Yes, we beat revenue guidance." has "direct" in train and "direct" in test -> agreed
        self.assertEqual(report.train_test_label_agreement_rate, 1.0)
        self.assertIn("Question-level leakage is strictly zero", report.methodological_note)

    def test_bootstrap_significance_test(self):
        y_true = ["direct"] * 50 + ["intermediate"] * 40 + ["fully_evasive"] * 10
        pred_m1 = ["direct"] * 45 + ["intermediate"] * 40 + ["intermediate"] * 15
        pred_m2 = ["direct"] * 48 + ["intermediate"] * 42 + ["fully_evasive"] * 10

        boot_res = bootstrap_significance_test(
            y_true=y_true,
            predictions={"m1": pred_m1, "m2": pred_m2},
            n_bootstraps=100,
            seed=42,
        )

        self.assertIn("models", boot_res)
        self.assertIn("m1", boot_res["models"])
        self.assertIn("m2", boot_res["models"])
        self.assertIn("pairwise_differences", boot_res)
        self.assertIn("m2_vs_m1", boot_res["pairwise_differences"])

        diff = boot_res["pairwise_differences"]["m2_vs_m1"]
        self.assertIn("macro_f1_delta_mean", diff)
        self.assertIn("macro_f1_delta_ci95", diff)
        self.assertIn("macro_f1_p_value_m2_le_m1", diff)
        self.assertIn("fully_evasive_f1_delta_mean", diff)

    def test_shared_splits_and_fair_baseline(self):
        parquet_path = Path("data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet")
        if not parquet_path.is_file():
            self.skipTest("EvasionBench Parquet dataset not found locally.")

        bundle = load_and_split_evasion_data(parquet_path=parquet_path, random_state=42)
        meta = bundle["metadata"]
        self.assertEqual(meta["train_samples"], 10704)
        self.assertEqual(meta["val_samples"], 2676)
        self.assertEqual(meta["test_samples"], 3346)
        self.assertEqual(meta["group_overlap_train_val"], 0)
        self.assertEqual(meta["group_overlap_train_test"], 0)
        self.assertEqual(meta["group_overlap_val_test"], 0)

        # Quick small-scale check on fair tfidf baseline function with small subset
        small_train = {
            "questions": bundle["train"]["questions"][:60],
            "answers": bundle["train"]["answers"][:60],
            "labels": bundle["train"]["labels"][:60],
        }
        small_test = {
            "questions": bundle["test"]["questions"][:30],
            "answers": bundle["test"]["answers"][:30],
            "labels": bundle["test"]["labels"][:30],
        }
        res = train_fair_tfidf_baseline(small_train, small_test, random_state=42)
        self.assertEqual(res["train_samples"], 60)
        self.assertEqual(res["test_samples"], 30)
        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)


if __name__ == "__main__":
    unittest.main()
