"""
FinVeritas: Audit Human-Annotated Model Outputs in FinanceBench.

Downloads and inspects model completion outputs and human evaluation annotations
from the official FinanceBench repository (results/).
Analyzes schemas, human correctness labels, refusal rates, duplicate records,
and evaluation-readiness for downstream claim/answer verification.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("audit_model_results")

GITHUB_API_RESULTS = "https://api.github.com/repos/patronus-ai/financebench/contents/results"
GITHUB_RAW_RESULTS = "https://raw.githubusercontent.com/patronus-ai/financebench/main/results"


def download_results_files(output_dir: Path) -> list[Path]:
    """Downloads all JSONL result files from the GitHub repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        GITHUB_API_RESULTS,
        headers={"User-Agent": "FinVeritas-Research/0.1.0"},
    )
    with urllib.request.urlopen(req) as resp:
        contents = json.loads(resp.read().decode("utf-8"))

    downloaded = []
    for item in contents:
        name = item["name"]
        if not name.endswith(".jsonl"):
            continue
        dest_path = output_dir / name
        if not (dest_path.is_file() and dest_path.stat().st_size > 0):
            dl_url = item.get("download_url") or f"{GITHUB_RAW_RESULTS}/{name}"
            logger.info(f"Downloading {name}...")
            file_req = urllib.request.Request(
                dl_url,
                headers={"User-Agent": "FinVeritas-Research/0.1.0"},
            )
            with urllib.request.urlopen(file_req) as file_resp:
                dest_path.write_bytes(file_resp.read())
        downloaded.append(dest_path)

    return downloaded


def audit_results_dataset(results_dir: Path) -> dict[str, Any]:
    """Inspects all results files, schema fields, human annotations, and consistency."""
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {results_dir}")

    file_summaries = []
    all_question_ids = set()
    all_schemas = {}
    label_distribution = Counter()
    total_records = 0
    duplicate_records = 0
    records_with_missing_labels = 0

    question_to_models: dict[str, list[str]] = {}

    for file_path in files:
        model_config = file_path.stem
        rows = []
        seen_in_file = set()
        file_dup = 0
        file_missing_label = 0
        file_label_dist = Counter()

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                total_records += 1

                # Track schema keys
                keys_tuple = tuple(sorted(data.keys()))
                all_schemas[keys_tuple] = all_schemas.get(keys_tuple, 0) + 1

                fb_id = data.get("financebench_id") or data.get("id") or f"row_{line_idx}"
                all_question_ids.add(fb_id)

                if fb_id not in question_to_models:
                    question_to_models[fb_id] = []
                question_to_models[fb_id].append(model_config)

                if fb_id in seen_in_file:
                    duplicate_records += 1
                    file_dup += 1
                seen_in_file.add(fb_id)

                # Look for human evaluation / correctness label fields
                label_val = None
                label_candidates = [
                    "label",
                    "correctness",
                    "human_label",
                    "score",
                    "is_correct",
                    "evaluation",
                ]
                for candidate in label_candidates:
                    if candidate in data:
                        label_val = data[candidate]
                        break

                if label_val is None:
                    records_with_missing_labels += 1
                    file_missing_label += 1
                    label_key = "MISSING_LABEL"
                else:
                    label_key = str(label_val).strip()

                label_distribution[label_key] += 1
                file_label_dist[label_key] += 1
                rows.append(data)

        file_summaries.append({
            "filename": file_path.name,
            "model_config": model_config,
            "row_count": len(rows),
            "unique_questions": len(seen_in_file),
            "duplicates": file_dup,
            "missing_labels": file_missing_label,
            "label_distribution": dict(file_label_dist),
            "sample_keys": list(rows[0].keys()) if rows else [],
        })

    # Multi-model answering analysis
    models_per_question = [len(m_list) for m_list in question_to_models.values()]
    avg_models = (
        sum(models_per_question) / len(models_per_question) if models_per_question else 0
    )

    return {
        "overview": {
            "total_files": len(files),
            "total_records": total_records,
            "distinct_model_configs": len(files),
            "distinct_question_ids": len(all_question_ids),
            "total_duplicates": duplicate_records,
            "total_missing_labels": records_with_missing_labels,
            "overall_label_distribution": dict(label_distribution),
            "avg_model_evaluations_per_question": round(avg_models, 2),
            "min_model_evaluations_per_question": (
                min(models_per_question) if models_per_question else 0
            ),
            "max_model_evaluations_per_question": (
                max(models_per_question) if models_per_question else 0
            ),
        },
        "file_details": file_summaries,
        "schema_variants": [
            {"keys": list(k), "count": cnt} for k, cnt in all_schemas.items()
        ],
    }


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    results_dir = Path("data/raw/financebench/results")

    logger.info("Downloading FinanceBench model results...")
    download_results_files(results_dir)

    logger.info("Auditing results data...")
    report = audit_results_dataset(results_dir)

    print("\n" + "=" * 70)
    print("FINANCEBENCH HUMAN-ANNOTATED MODEL OUTPUTS AUDIT")
    print("=" * 70)
    ov = report["overview"]
    print(f"Total Results Files:             {ov['total_files']}")
    print(f"Total Model Completion Records:  {ov['total_records']}")
    print(f"Distinct Model Configurations:   {ov['distinct_model_configs']}")
    print(f"Distinct Benchmark Questions:    {ov['distinct_question_ids']}")
    print(f"Duplicate Records:               {ov['total_duplicates']}")
    print(f"Missing Labels:                  {ov['total_missing_labels']}")
    print(f"Avg Models per Question:         {ov['avg_model_evaluations_per_question']}")
    print("\nOverall Label Distribution:")
    for lbl, cnt in ov["overall_label_distribution"].items():
        print(f"  - {lbl:<25}: {cnt:4d} ({cnt/ov['total_records']*100:5.1f}%)")

    print("\nFile-by-File Breakdown:")
    for f in report["file_details"]:
        print(
            f"  {f['model_config']:<35} | {f['row_count']:3d} rows | "
            f"Labels: {f['label_distribution']}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
