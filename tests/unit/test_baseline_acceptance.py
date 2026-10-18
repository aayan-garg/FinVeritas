"""
FinVeritas Acceptance Tests for EvasionBench Data & Baseline Pipeline
Verifies schema compliance, class weight calculation, memory safety bounds, and metrics.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add repo root to sys.path so scripts can be imported cleanly
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.train_baseline import compute_class_weights, evaluate_predictions, LABEL2ID, ID2LABEL, get_training_args

class TestFinVeritasBaseline(unittest.TestCase):
    
    def test_label_mapping(self):
        self.assertEqual(len(LABEL2ID), 3)
        self.assertEqual(LABEL2ID["direct"], 0)
        self.assertEqual(LABEL2ID["intermediate"], 1)
        self.assertEqual(LABEL2ID["fully_evasive"], 2)
        for k, v in LABEL2ID.items():
            self.assertEqual(ID2LABEL[v], k)

    def test_compute_class_weights_imbalance(self):
        # Simulate HF distribution: 52% direct, 44% intermediate, 4% fully_evasive
        labels = ["direct"] * 520 + ["intermediate"] * 440 + ["fully_evasive"] * 40
        weights = compute_class_weights(labels)
        
        self.assertEqual(weights.shape[0], 3)
        # Minority class (fully_evasive) must have substantially higher weight
        self.assertGreater(weights[2].item(), weights[0].item() * 5.0)
        self.assertGreater(weights[2].item(), weights[1].item() * 5.0)
        print(f"\n[Test] Computed Class Weights for Simulated Imbalance: {weights.tolist()}")

    def test_evaluate_predictions_metrics(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 0]  # One misclassification on class 2
        metrics = evaluate_predictions(y_true, y_pred)
        
        self.assertIn("macro_f1", metrics)
        self.assertIn("direct_f1", metrics)
        self.assertIn("intermediate_f1", metrics)
        self.assertIn("fully_evasive_f1", metrics)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["intermediate_f1"], 1.0)

    def test_training_args_memory_bounds(self):
        args = get_training_args()
        # VRAM safety assertions for 6GB RTX 3050 (usable budget ~4.9GB):
        # Batch size <= 16, max sequence length <= 384
        self.assertLessEqual(args["batch_size"], 16, "Batch size exceeds safe 6GB VRAM threshold")
        self.assertLessEqual(args["max_length"], 384, "Sequence length exceeds safe 6GB VRAM threshold")
        self.assertEqual(args["grad_accum_steps"], 2, "Effective batch size must equal 32 via gradient accumulation")

if __name__ == "__main__":
    unittest.main()
