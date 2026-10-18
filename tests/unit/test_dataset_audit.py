"""
FinVeritas: Offline Unit Tests for EvasionBench Dataset Auditor.
Tests schema validation, label checks, duplicates, blank handling,
letter mappings, SHA-256 computation, and CLI offline behavior
using small synthetic PyArrow Parquet fixtures.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from finveritas.nlp_evasion.dataset_audit import (
    audit_parquet_dataset,
    compute_sha256,
    download_evasionbench_parquet,
)


class TestDatasetAudit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_parquet(
        self,
        data: dict,
        filename: str = "synthetic.parquet",
    ) -> Path:
        file_path = self.temp_path / filename
        table = pa.Table.from_pydict(data)
        pq.write_table(table, file_path)
        return file_path

    def test_sha256_calculation(self):
        test_file = self.temp_path / "test_bytes.bin"
        test_file.write_bytes(b"FinVeritas reproducible audit SHA-256 test")
        computed = compute_sha256(test_file)
        # hashlib.sha256(b"FinVeritas reproducible audit SHA-256 test").hexdigest()
        expected = "7cd729331f7b7a160db8f2e440bb477f96b61544058443e43750056b15070b7e"
        self.assertEqual(computed, expected)

    def test_valid_schema_and_class_distribution(self):
        data = {
            "uid": ["u1", "u2", "u3", "u4"],
            "question": [
                "What was Q3 revenue?",
                "Will margins expand?",
                "Why did churn rise?",
                "Guidance?",
            ],
            "answer": [
                "Revenue was $5B.",
                "We expect slight expansion.",
                "Due to pricing.",
                "We do not guide.",
            ],
            "eva4b_label": ["direct", "intermediate", "direct", "fully_evasive"],
            "choices": [
                ["direct", "intermediate", "fully_evasive"],
                ["direct", "intermediate", "fully_evasive"],
                ["direct", "intermediate", "fully_evasive"],
                ["direct", "intermediate", "fully_evasive"],
            ],
            "eva4b_label_letter": ["A", "B", "A", "C"],
        }
        parquet_path = self._create_synthetic_parquet(data)
        result = audit_parquet_dataset(parquet_path)

        self.assertEqual(result.row_count, 4)
        self.assertEqual(len(result.validation_errors), 0)
        self.assertTrue(result.is_model_generated_pseudo_label)

        # Class counts: 2 direct (50%), 1 intermediate (25%), 1 fully_evasive (25%)
        self.assertEqual(result.class_distribution["direct"]["count"], 2)
        self.assertEqual(result.class_distribution["direct"]["percentage"], 50.0)
        self.assertEqual(result.class_distribution["intermediate"]["count"], 1)
        self.assertEqual(result.class_distribution["intermediate"]["percentage"], 25.0)
        self.assertEqual(result.class_distribution["fully_evasive"]["count"], 1)
        self.assertEqual(result.class_distribution["fully_evasive"]["percentage"], 25.0)

        # Text length metrics
        self.assertGreater(result.question_length_stats["max"], 0)
        self.assertGreater(result.answer_length_stats["max"], 0)

        # Letter mapping
        self.assertTrue(result.letter_mapping_validation["mapping_valid"])
        self.assertEqual(result.letter_mapping_validation["mismatches"], 0)

    def test_missing_required_columns(self):
        # Missing 'eva4b_label' and 'answer'
        data = {
            "uid": ["u1", "u2"],
            "question": ["Q1", "Q2"],
        }
        parquet_path = self._create_synthetic_parquet(data)
        result = audit_parquet_dataset(parquet_path)

        self.assertIn("Missing required column: 'answer'", result.validation_errors)
        self.assertIn("Missing required column: 'eva4b_label'", result.validation_errors)

    def test_null_and_blank_text_accounting(self):
        data = {
            "uid": ["u1", "u2", "u3"],
            "question": ["Valid question", "   ", None],
            "answer": ["Valid answer", None, ""],
            "eva4b_label": ["direct", "intermediate", "fully_evasive"],
        }
        parquet_path = self._create_synthetic_parquet(data)
        result = audit_parquet_dataset(parquet_path)

        self.assertEqual(result.missing_and_blank_counts["question"]["null_count"], 1)
        self.assertEqual(result.missing_and_blank_counts["question"]["blank_count"], 1)
        self.assertEqual(result.missing_and_blank_counts["answer"]["null_count"], 1)
        self.assertEqual(result.missing_and_blank_counts["answer"]["blank_count"], 1)

    def test_duplicate_detection(self):
        data = {
            "uid": ["u1", "u1", "u2", "u3"],  # 1 duplicate UID
            "question": ["Same Q", "Same Q", "Same Q", "Unique Q"],  # 2 duplicate Qs
            "answer": ["Same A", "Same A", "Diff A", "Unique A"],  # 1 duplicate A
            "eva4b_label": ["direct", "direct", "intermediate", "fully_evasive"],
        }
        parquet_path = self._create_synthetic_parquet(data)
        result = audit_parquet_dataset(parquet_path)

        self.assertEqual(result.duplicate_statistics["duplicate_uids"], 1)
        self.assertEqual(result.duplicate_statistics["unique_uids"], 3)
        self.assertEqual(result.duplicate_statistics["duplicate_questions"], 2)
        self.assertEqual(result.duplicate_statistics["unique_questions"], 2)
        self.assertEqual(result.duplicate_statistics["duplicate_answers"], 1)
        self.assertEqual(result.duplicate_statistics["duplicate_qa_pairs"], 1)

    def test_invalid_labels(self):
        data = {
            "uid": ["u1", "u2"],
            "question": ["Q1", "Q2"],
            "answer": ["A1", "A2"],
            "eva4b_label": ["direct", "completely_unsupported_label"],
        }
        parquet_path = self._create_synthetic_parquet(data)
        result = audit_parquet_dataset(parquet_path)

        self.assertTrue(
            any("Unexpected label found in eva4b_label" in err for err in result.validation_errors)
        )

    def test_letter_mapping_mismatch(self):
        data = {
            "uid": ["u1"],
            "question": ["Q1"],
            "answer": ["A1"],
            "eva4b_label": ["direct"],
            "choices": [["direct", "intermediate", "fully_evasive"]],
            "eva4b_label_letter": ["B"],  # Letter 'B' maps to choices[1]='intermediate' != 'direct'
        }
        parquet_path = self._create_synthetic_parquet(data)
        result = audit_parquet_dataset(parquet_path)

        self.assertFalse(result.letter_mapping_validation["mapping_valid"])
        self.assertEqual(result.letter_mapping_validation["mismatches"], 1)
        self.assertTrue(
            any("eva4b_label_letter mismatches detected" in err for err in result.validation_errors)
        )

    def test_json_serialization(self):
        data = {
            "uid": ["u1"],
            "question": ["Question?"],
            "answer": ["Answer."],
            "eva4b_label": ["direct"],
        }
        parquet_path = self._create_synthetic_parquet(data)
        result = audit_parquet_dataset(parquet_path)

        json_str = result.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["filename"], "synthetic.parquet")
        self.assertEqual(parsed["row_count"], 1)
        self.assertEqual(parsed["duplicate_statistics"]["total_rows"], 1)

    def test_cli_help_execution_without_network(self):
        cmd = [sys.executable, "-m", "finveritas.nlp_evasion.dataset_audit", "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("FinVeritas EvasionBench Dataset Auditor", proc.stdout)
        self.assertIn("--parquet-path", proc.stdout)
        self.assertIn("--download", proc.stdout)

    def test_offline_local_file_audit_via_cli(self):
        data = {
            "uid": ["u1"],
            "question": ["Sample question?"],
            "answer": ["Sample answer."],
            "eva4b_label": ["direct"],
        }
        parquet_path = self._create_synthetic_parquet(data, filename="cli_test.parquet")
        out_json = self.temp_path / "cli_audit.json"

        cmd = [
            sys.executable,
            "-m",
            "finveritas.nlp_evasion.dataset_audit",
            "--parquet-path",
            str(parquet_path),
            "--output-json",
            str(out_json),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI failed: {proc.stderr}")
        self.assertTrue(out_json.is_file())
        with open(out_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["row_count"], 1)

    def test_download_safeguard_insecure_url(self):
        with self.assertRaises(ValueError) as ctx:
            download_evasionbench_parquet(
                "http://insecure.example.com/data.parquet",
                self.temp_path / "test.parquet",
            )
        self.assertIn("HTTPS is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
