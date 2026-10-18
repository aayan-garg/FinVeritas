"""
FinVeritas: Unit Tests for Transformer Classifier.
Tests dataset tokenization, dynamic padding, truncation tracking,
normalized-question group disjoint partitions, metrics computation,
and single-pair inference contracts.
"""

import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finveritas.nlp_evasion.transformer_classifier import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DISCLAIMER_TEXT,
    SUPPORTED_CLASSES,
    EvasionQAPairDataset,
    collate_qa_batch,
    compute_metrics,
    load_and_split_evasion_data,
    predict_evasion_transformer,
    run_forward_backward_smoke_test,
)


class TestTransformerClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    def test_evasion_qa_dataset_and_truncation(self):
        questions = [
            "What was Q3 revenue?",
            "Can you discuss forward margin expectations?",
        ]
        answers = [
            "Revenue reached $5.2B for the quarter.",
            "We decline to comment on future margins.",
        ]
        labels = ["direct", "fully_evasive"]

        ds = EvasionQAPairDataset(
            questions=questions,
            answers=answers,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=64,
        )

        self.assertEqual(len(ds), 2)
        item0 = ds[0]
        self.assertIn("input_ids", item0)
        self.assertIn("attention_mask", item0)
        self.assertIn("labels", item0)
        self.assertEqual(item0["labels"].item(), 0)  # "direct"

        item1 = ds[1]
        self.assertEqual(item1["labels"].item(), 2)  # "fully_evasive"

        # Check truncation stats on short max_length
        ds_short = EvasionQAPairDataset(
            questions=["Question: " + ("word " * 50)],
            answers=["Answer: " + ("answer " * 50)],
            labels=["direct"],
            tokenizer=self.tokenizer,
            max_length=16,
        )
        self.assertEqual(ds_short.truncation_stats.truncated_samples, 1)
        self.assertEqual(ds_short.truncation_stats.truncation_rate, 1.0)

    def test_collate_qa_batch(self):
        item1 = {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor(0, dtype=torch.long),
        }
        item2 = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor(1, dtype=torch.long),
        }

        batch = collate_qa_batch([item1, item2], pad_token_id=0)
        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["attention_mask"].shape, (2, 5))
        self.assertEqual(batch["labels"].shape, (2,))
        # First item should be padded to 5 tokens
        self.assertEqual(batch["input_ids"][0, 3].item(), 0)
        self.assertEqual(batch["attention_mask"][0, 3].item(), 0)

    def test_load_and_split_evasion_data_no_overlap(self):
        parquet_path = Path("data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet")
        if not parquet_path.is_file():
            self.skipTest("EvasionBench Parquet dataset not found locally.")

        bundle = load_and_split_evasion_data(parquet_path=parquet_path, random_state=42)
        meta = bundle["metadata"]

        self.assertEqual(meta["total_samples"], 16726)
        self.assertEqual(meta["test_samples"], 3346)
        self.assertEqual(meta["train_samples"], 10704)
        self.assertEqual(meta["val_samples"], 2676)
        self.assertEqual(meta["train_samples"] + meta["val_samples"], 13380)

        # Zero question-group overlap assertions
        self.assertEqual(meta["group_overlap_train_val"], 0)
        self.assertEqual(meta["group_overlap_train_test"], 0)
        self.assertEqual(meta["group_overlap_val_test"], 0)

        # Class weights present for all 3 classes
        class_weights = bundle["class_weights"]
        for cls_name in SUPPORTED_CLASSES:
            self.assertIn(cls_name, class_weights)
            self.assertGreater(class_weights[cls_name], 0.0)

    def test_compute_metrics(self):
        y_true = ["direct", "direct", "intermediate", "fully_evasive"]
        y_pred = ["direct", "intermediate", "intermediate", "fully_evasive"]

        metrics = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertGreater(metrics["macro_f1"], 0.0)
        self.assertGreater(metrics["weighted_f1"], 0.0)
        self.assertIn("per_class", metrics)
        for cls_name in SUPPORTED_CLASSES:
            self.assertIn(cls_name, metrics["per_class"])
        self.assertEqual(len(metrics["confusion_matrix"]), 3)

    def test_forward_backward_smoke_test(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        res = run_forward_backward_smoke_test(
            model_name=DEFAULT_MODEL_NAME,
            device=device,
            max_length=64,
        )
        self.assertTrue(res["smoke_test_passed"])
        self.assertGreater(res["loss"], 0.0)

    def test_predict_evasion_transformer_contract(self):
        from transformers import AutoModelForSequenceClassification

        device = torch.device("cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            DEFAULT_MODEL_NAME,
            num_labels=len(SUPPORTED_CLASSES),
        )
        model.to(device)

        res = predict_evasion_transformer(
            model=model,
            tokenizer=self.tokenizer,
            question="What was Q3 revenue?",
            answer="Revenue grew 10% to $3.5B.",
            device=device,
            max_length=DEFAULT_MAX_LENGTH,
        )

        self.assertIn("predicted_label", res)
        self.assertIn(res["predicted_label"], SUPPORTED_CLASSES)
        self.assertIn("uncalibrated_probabilities", res)
        probs = res["uncalibrated_probabilities"]
        for cls_name in SUPPORTED_CLASSES:
            self.assertIn(cls_name, probs)
            self.assertGreaterEqual(probs[cls_name], 0.0)
            self.assertLessEqual(probs[cls_name], 1.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
        self.assertIn("disclaimer", res)
        self.assertEqual(res["disclaimer"], DISCLAIMER_TEXT)


if __name__ == "__main__":
    unittest.main()
