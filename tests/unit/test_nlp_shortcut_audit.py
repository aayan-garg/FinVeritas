"""
FinVeritas: Unit Tests for NLP Shortcut Learning and Robustness Audit.
Tests:
1. Strict normalized-answer-disjoint split creation and zero answer leakage.
2. Answer template extraction, frequency counting, and label purity.
3. Subset evaluation for overlapping vs unseen answer cohorts.
4. EvasionQAPairDataset input modes ('qa', 'question_only', 'answer_only').
5. Truncation and collation under single-sequence input modes.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finveritas.nlp_evasion.shortcut_audit import (
    audit_answer_templates,
    create_answer_disjoint_splits,
    evaluate_answer_template_subsets,
)
from finveritas.nlp_evasion.transformer_classifier import (
    DEFAULT_MODEL_NAME,
    EvasionQAPairDataset,
    collate_qa_batch,
)


class TestNLPShortcutAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    def test_evasion_dataset_input_modes(self):
        questions = [
            "What was Q3 revenue and operating profit margin?",
            "Can you discuss forward market share expectations?",
        ]
        answers = [
            "Revenue reached $5.2B for the quarter and margins expanded 120bps.",
            "We decline to comment on future market share.",
        ]
        labels = ["direct", "fully_evasive"]

        # 1. QA pair mode
        ds_qa = EvasionQAPairDataset(
            questions=questions,
            answers=answers,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=128,
            input_mode="qa",
        )
        item_qa = ds_qa[0]
        self.assertIn("input_ids", item_qa)
        self.assertIn("attention_mask", item_qa)
        self.assertIn("labels", item_qa)
        self.assertEqual(item_qa["labels"].item(), 0)

        # 2. Question-only mode
        ds_q = EvasionQAPairDataset(
            questions=questions,
            answers=answers,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=128,
            input_mode="question_only",
        )
        item_q = ds_q[0]
        # In question-only mode, sequence length should be strictly less than QA pair
        self.assertLess(item_q["input_ids"].size(0), item_qa["input_ids"].size(0))

        # 3. Answer-only mode
        ds_a = EvasionQAPairDataset(
            questions=questions,
            answers=answers,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=128,
            input_mode="answer_only",
        )
        item_a = ds_a[0]
        self.assertIn("input_ids", item_a)
        self.assertIn("labels", item_a)

        # 4. Invalid mode raises ValueError
        with self.assertRaises(ValueError):
            EvasionQAPairDataset(
                questions=questions,
                answers=answers,
                labels=labels,
                tokenizer=self.tokenizer,
                input_mode="invalid_mode",
            )

    def test_collate_single_sequence_batch(self):
        item1 = {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor(0, dtype=torch.long),
        }
        item2 = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor(1, dtype=torch.long),
        }

        batch = collate_qa_batch([item1, item2], pad_token_id=0)
        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["attention_mask"].shape, (2, 6))
        self.assertEqual(batch["labels"].shape, (2,))
        self.assertEqual(batch["input_ids"][0, 3].item(), 0)

    def test_audit_answer_templates(self):
        answers = [
            "That's correct.",
            "That's correct.",
            "That's correct.",
            "We decline to comment.",
            "We decline to comment.",
            "Custom unique answer 12345.",
        ]
        labels = [
            "direct",
            "direct",
            "intermediate",
            "fully_evasive",
            "fully_evasive",
            "direct",
        ]

        templates = audit_answer_templates(answers, labels, top_n=5)
        self.assertEqual(len(templates), 2)  # Only answers with count > 1

        top1 = templates[0]
        self.assertEqual(top1["template"], "that's correct.")
        self.assertEqual(top1["total_count"], 3)
        self.assertEqual(top1["dominant_label"], "direct")
        self.assertEqual(top1["label_distribution"]["direct"], 2)
        self.assertEqual(top1["label_distribution"]["intermediate"], 1)
        self.assertAlmostEqual(top1["dominant_label_purity"], 2 / 3, places=3)

        top2 = templates[1]
        self.assertEqual(top2["template"], "we decline to comment.")
        self.assertEqual(top2["total_count"], 2)
        self.assertEqual(top2["dominant_label"], "fully_evasive")
        self.assertEqual(top2["dominant_label_purity"], 1.0)

    def test_evaluate_answer_template_subsets(self):
        y_true = ["direct", "direct", "intermediate", "fully_evasive"]
        y_pred = ["direct", "intermediate", "intermediate", "fully_evasive"]
        is_overlapping = [True, True, False, False]

        subsets = evaluate_answer_template_subsets(y_true, y_pred, is_overlapping)
        self.assertEqual(subsets["overlapping_answers"]["sample_count"], 2)
        self.assertEqual(subsets["unseen_answers"]["sample_count"], 2)
        self.assertEqual(subsets["overlapping_answers"]["accuracy"], 0.5)
        self.assertEqual(subsets["unseen_answers"]["accuracy"], 1.0)
        self.assertEqual(subsets["overall"]["accuracy"], 0.75)

    def test_answer_disjoint_split_zero_leakage(self):
        parquet_path = Path("data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet")
        if not parquet_path.is_file():
            self.skipTest("EvasionBench Parquet dataset not found locally.")

        splits = create_answer_disjoint_splits(parquet_path=parquet_path, random_state=42)
        meta = splits["metadata"]

        self.assertEqual(meta["partitioning_type"], "normalized_answer_disjoint")
        self.assertEqual(meta["train_samples"], 10704)
        self.assertEqual(meta["val_samples"], 2676)
        self.assertEqual(meta["test_samples"], 3346)
        self.assertEqual(meta["answer_group_overlap_train_val"], 0)
        self.assertEqual(meta["answer_group_overlap_train_test"], 0)
        self.assertEqual(meta["answer_group_overlap_val_test"], 0)

        # Verify class weights computed strictly from training partition
        weights = splits["class_weights"]
        self.assertIn("direct", weights)
        self.assertIn("intermediate", weights)
        self.assertIn("fully_evasive", weights)
        self.assertGreater(weights["fully_evasive"], weights["direct"])

    def test_shortcut_audit_metrics_consistency_and_report_integrity(self):
        metrics_file = Path("artifacts/nlp_shortcut_audit/metrics.json")
        if not metrics_file.is_file():
            self.skipTest("artifacts/nlp_shortcut_audit/metrics.json not found.")

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        # 1. Verify input ablation metrics consistency
        ablations = metrics_data["experiment_1_input_ablations"]
        qa_metrics = ablations["model_c_question_and_answer_384"]
        a_metrics = ablations["model_b_answer_only_384"]
        q_metrics = ablations["model_a_question_only_384"]

        # Question-only minority class collapse on this setup
        self.assertEqual(q_metrics["per_class"]["fully_evasive"]["recall"], 0.0)
        self.assertEqual(q_metrics["per_class"]["fully_evasive"]["f1"], 0.0)
        # But non-zero on direct and intermediate
        self.assertGreater(q_metrics["per_class"]["direct"]["f1"], 0.60)
        self.assertGreater(q_metrics["per_class"]["intermediate"]["f1"], 0.60)

        # Answer-only macro-F1 is higher than QA macro-F1
        self.assertGreater(a_metrics["macro_f1"], qa_metrics["macro_f1"])

        # QA fully-evasive recall is higher than answer-only fully-evasive recall (trade-off)
        self.assertGreater(
            qa_metrics["per_class"]["fully_evasive"]["recall"],
            a_metrics["per_class"]["fully_evasive"]["recall"],
        )
        self.assertAlmostEqual(
            qa_metrics["per_class"]["fully_evasive"]["recall"], 78 / 124, places=3
        )
        self.assertAlmostEqual(
            a_metrics["per_class"]["fully_evasive"]["recall"], 61 / 124, places=3
        )

        # 2. Verify report file exists and contains scientific qualifications
        report_path = Path("docs/research/nlp_shortcut_audit.md")
        self.assertTrue(report_path.is_file())
        report_text = report_path.read_text(encoding="utf-8")

        # Discredited phrasing must not be present
        self.assertNotIn("acts as a distractor during fine-tuning", report_text)
        self.assertNotIn("The Question Contains Zero Evasion Signal", report_text)
        self.assertNotIn("accounts for approximately 10.9 F1 points", report_text)

        # Required scientific interpretations must be present
        expected_tradeoff = (
            "trade-off between aggregate classification performance and "
            "evasive-example sensitivity"
        )
        self.assertIn(expected_tradeoff, report_text)
        self.assertIn("distinct evaluation populations", report_text)
        self.assertIn("pseudo-labels", report_text)

        # 3. Verify errata document exists
        errata_path = Path("docs/research/nlp_shortcut_audit_errata.md")
        self.assertTrue(errata_path.is_file())

