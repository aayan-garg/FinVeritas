"""
FinVeritas: Reproducible, Resource-Efficient EvasionBench Dataset Auditor.

Audits schema, provenance, class distribution, duplicate patterns, text-length
metrics, and group/temporal leakage indicators in bounded memory without
requiring GPU, PyTorch, Transformers, or pandas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

SUPPORTED_CLASSES = ("direct", "intermediate", "fully_evasive")
REQUIRED_COLUMNS = ("uid", "question", "answer", "eva4b_label")
GROUP_TEMPORAL_COLUMNS = ("ticker", "company", "call_date", "transcript_id")
MAX_DOWNLOAD_SIZE_BYTES = 32 * 1024 * 1024  # 32 MB safeguard


@dataclass
class TextLengthStats:
    min: int = 0
    median: float = 0.0
    p95: float = 0.0
    max: int = 0


@dataclass
class DatasetAuditResult:
    dataset_name: str
    source_url: str | None
    observed_revision: str | None
    file_path: str
    filename: str
    file_size_bytes: int
    sha256_hash: str
    row_count: int
    row_group_count: int
    schema_fields: dict[str, str]
    label_provenance: str
    is_model_generated_pseudo_label: bool
    class_distribution: dict[str, dict[str, Any]]
    missing_and_blank_counts: dict[str, dict[str, int]]
    duplicate_statistics: dict[str, int]
    question_length_stats: dict[str, Any]
    answer_length_stats: dict[str, Any]
    letter_mapping_validation: dict[str, Any]
    group_temporal_metadata: dict[str, Any]
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash in 64KB bounded chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_length_stats(lengths: list[int]) -> TextLengthStats:
    """Computes min, median, 95th percentile, and max for character lengths."""
    if not lengths:
        return TextLengthStats()
    sorted_len = sorted(lengths)
    n = len(sorted_len)
    p95_idx = int(0.95 * (n - 1))
    return TextLengthStats(
        min=int(sorted_len[0]),
        median=float(statistics.median(sorted_len)),
        p95=float(sorted_len[p95_idx]),
        max=int(sorted_len[-1]),
    )


def download_evasionbench_parquet(
    url: str,
    target_path: Path,
    max_size_bytes: int = MAX_DOWNLOAD_SIZE_BYTES,
    timeout_seconds: int = 30,
) -> Path:
    """
    Downloads the approved Parquet file using HTTPS, request timeouts, and
    an atomic rename of a temporary file, with strict file-size bounds.
    """
    if not url.startswith("https://"):
        raise ValueError(f"Insecure or invalid URL: {url}. HTTPS is required.")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # If already present, verify and return
    if target_path.is_file():
        logger.info(f"File already exists at {target_path}. Skipping download.")
        return target_path

    logger.info(f"Connecting to {url} (timeout={timeout_seconds}s)...")
    req = urllib.request.Request(url, headers={"User-Agent": "FinVeritas-Audit/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                content_length_int = int(content_length)
                if content_length_int > max_size_bytes:
                    raise ValueError(
                        f"Content-Length {content_length_int} bytes exceeds "
                        f"maximum allowed limit of {max_size_bytes} bytes."
                    )

            # Download to temporary file in same directory for atomic replace
            with tempfile.NamedTemporaryFile(
                dir=target_path.parent, delete=False, suffix=".tmp"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                total_downloaded = 0
                chunk_size = 65536

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    total_downloaded += len(chunk)
                    if total_downloaded > max_size_bytes:
                        tmp_file.close()
                        tmp_path.unlink(missing_ok=True)
                        raise ValueError(
                            f"Downloaded data exceeded maximum limit of {max_size_bytes} bytes."
                        )
                    tmp_file.write(chunk)

        # Atomic replacement
        tmp_path.replace(target_path)
        logger.info(
            f"Successfully downloaded {target_path.name} "
            f"({target_path.stat().st_size} bytes)."
        )
        return target_path

    except Exception as e:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download dataset from {url}: {e}") from e


def audit_parquet_dataset(
    parquet_path: Path,
    config: dict[str, Any] | None = None,
) -> DatasetAuditResult:
    """
    Performs a full, resource-efficient, bounded-memory audit of a local Parquet file.
    Does not access network resources.
    """
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")

    config = config or {}
    dataset_name = config.get("dataset_name", "EvasionBench")
    source_url = config.get("source_url")
    observed_revision = config.get("hf_revision")
    file_size = parquet_path.stat().st_size
    sha256 = compute_sha256(parquet_path)

    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as err:
        raise ValueError(f"Corrupt or invalid Parquet file at {parquet_path}: {err}") from err

    metadata = parquet_file.metadata
    num_rows = metadata.num_rows
    num_row_groups = metadata.num_row_groups
    schema = parquet_file.schema_arrow

    schema_fields = {field.name: str(field.type) for field in schema}

    validation_errors: list[str] = []
    validation_warnings: list[str] = []

    # Check required columns
    for req_col in REQUIRED_COLUMNS:
        if req_col not in schema_fields:
            validation_errors.append(f"Missing required column: '{req_col}'")

    # Check group/temporal columns
    present_group_cols = [col for col in GROUP_TEMPORAL_COLUMNS if col in schema_fields]
    absent_group_cols = [col for col in GROUP_TEMPORAL_COLUMNS if col not in schema_fields]

    group_temporal_metadata = {
        "present_identifiers": present_group_cols,
        "absent_identifiers": absent_group_cols,
        "supports_temporal_splitting": "call_date" in present_group_cols,
        "supports_group_aware_splitting": any(
            c in present_group_cols for c in ("ticker", "company", "transcript_id")
        ),
    }

    if absent_group_cols:
        validation_warnings.append(
            f"Missing identifiers for company-level/temporal leakage control: {absent_group_cols}"
        )

    # Streaming bounded-batch accumulators
    class_counts: Counter[str] = Counter()
    missing_and_blank: dict[str, dict[str, int]] = {
        col: {"null_count": 0, "blank_count": 0} for col in schema_fields
    }

    seen_uids: set[str] = set()
    dup_uid_count = 0

    seen_questions: set[bytes] = set()
    dup_question_count = 0

    seen_answers: set[bytes] = set()
    dup_answer_count = 0

    seen_qa_pairs: set[bytes] = set()
    dup_qa_pair_count = 0

    question_lengths: list[int] = []
    answer_lengths: list[int] = []

    has_choices = "choices" in schema_fields
    has_letter = "eva4b_label_letter" in schema_fields
    letter_mapping_mismatches = 0
    letter_mapping_total = 0

    batch_size = 2048
    columns_to_read = list(schema_fields.keys())

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns_to_read):
        batch_pydict = batch.to_pydict()

        for col, values in batch_pydict.items():
            for val in values:
                if val is None:
                    missing_and_blank[col]["null_count"] += 1
                elif isinstance(val, str) and not val.strip():
                    missing_and_blank[col]["blank_count"] += 1

        # UIDs
        if "uid" in batch_pydict:
            for uid in batch_pydict["uid"]:
                if uid is not None:
                    uid_str = str(uid)
                    if uid_str in seen_uids:
                        dup_uid_count += 1
                    else:
                        seen_uids.add(uid_str)

        # Questions
        if "question" in batch_pydict:
            for q in batch_pydict["question"]:
                if q is not None:
                    q_str = str(q)
                    question_lengths.append(len(q_str))
                    q_hash = hashlib.sha256(q_str.encode("utf-8")).digest()
                    if q_hash in seen_questions:
                        dup_question_count += 1
                    else:
                        seen_questions.add(q_hash)

        # Answers
        if "answer" in batch_pydict:
            for a in batch_pydict["answer"]:
                if a is not None:
                    a_str = str(a)
                    answer_lengths.append(len(a_str))
                    a_hash = hashlib.sha256(a_str.encode("utf-8")).digest()
                    if a_hash in seen_answers:
                        dup_answer_count += 1
                    else:
                        seen_answers.add(a_hash)

        # QA Pairs
        if "question" in batch_pydict and "answer" in batch_pydict:
            for q, a in zip(batch_pydict["question"], batch_pydict["answer"]):
                if q is not None and a is not None:
                    qa_str = f"{q}\n---\n{a}"
                    qa_hash = hashlib.sha256(qa_str.encode("utf-8")).digest()
                    if qa_hash in seen_qa_pairs:
                        dup_qa_pair_count += 1
                    else:
                        seen_qa_pairs.add(qa_hash)

        # Class counts
        if "eva4b_label" in batch_pydict:
            for lbl in batch_pydict["eva4b_label"]:
                if lbl is not None:
                    class_counts[str(lbl)] += 1

        # Choice / Letter validation
        if has_choices and has_letter:
            for choices, letter, lbl in zip(
                batch_pydict["choices"],
                batch_pydict["eva4b_label_letter"],
                batch_pydict["eva4b_label"],
            ):
                if letter is not None and choices is not None and lbl is not None:
                    letter_mapping_total += 1
                    letter_str = str(letter).strip().upper()
                    idx = ord(letter_str) - ord("A") if len(letter_str) == 1 else -1
                    if 0 <= idx < len(choices):
                        choice_val = choices[idx]
                        if str(choice_val) != str(lbl):
                            letter_mapping_mismatches += 1
                    else:
                        letter_mapping_mismatches += 1

    # Validate classes
    for c in class_counts:
        if c not in SUPPORTED_CLASSES:
            validation_errors.append(f"Unexpected label found in eva4b_label: '{c}'")

    class_distribution: dict[str, dict[str, Any]] = {}
    total_labels = sum(class_counts.values())
    for c in SUPPORTED_CLASSES:
        cnt = class_counts.get(c, 0)
        pct = (cnt / total_labels * 100.0) if total_labels > 0 else 0.0
        class_distribution[c] = {
            "count": cnt,
            "percentage": round(pct, 2),
        }

    # Check for severe imbalance
    if class_counts.get("fully_evasive", 0) < 0.1 * total_labels:
        validation_warnings.append(
            f"Severe class skew detected: 'fully_evasive' represents only "
            f"{class_distribution['fully_evasive']['percentage']}% of samples."
        )

    duplicate_stats = {
        "total_rows": num_rows,
        "unique_uids": len(seen_uids),
        "duplicate_uids": dup_uid_count,
        "unique_questions": len(seen_questions),
        "duplicate_questions": dup_question_count,
        "unique_answers": len(seen_answers),
        "duplicate_answers": dup_answer_count,
        "unique_qa_pairs": len(seen_qa_pairs),
        "duplicate_qa_pairs": dup_qa_pair_count,
    }

    q_length_stats = compute_length_stats(question_lengths)
    a_length_stats = compute_length_stats(answer_lengths)

    letter_validation = {
        "has_choices_column": has_choices,
        "has_letter_column": has_letter,
        "total_evaluated": letter_mapping_total,
        "mismatches": letter_mapping_mismatches,
        "mapping_valid": (letter_mapping_total > 0 and letter_mapping_mismatches == 0)
        if (has_choices and has_letter)
        else None,
    }

    if letter_mapping_mismatches > 0:
        validation_errors.append(
            f"eva4b_label_letter mismatches detected: "
            f"{letter_mapping_mismatches} of {letter_mapping_total}"
        )

    return DatasetAuditResult(
        dataset_name=dataset_name,
        source_url=source_url,
        observed_revision=observed_revision,
        file_path=str(parquet_path),
        filename=parquet_path.name,
        file_size_bytes=file_size,
        sha256_hash=sha256,
        row_count=num_rows,
        row_group_count=num_row_groups,
        schema_fields=schema_fields,
        label_provenance="Eva-4B-V2 model-generated pseudo-labels (arXiv:2601.09142)",
        is_model_generated_pseudo_label=True,
        class_distribution=class_distribution,
        missing_and_blank_counts=missing_and_blank,
        duplicate_statistics=duplicate_stats,
        question_length_stats=asdict(q_length_stats),
        answer_length_stats=asdict(a_length_stats),
        letter_mapping_validation=letter_validation,
        group_temporal_metadata=group_temporal_metadata,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
    )


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path and config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinVeritas EvasionBench Dataset Auditor (Offline-First)"
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet"),
        help="Path to the local Parquet file to audit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/evasionbench.yaml"),
        help="Path to YAML dataset configuration file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/dataset_audit/evasionbench_audit.json"),
        help="Path to write the machine-readable JSON audit report.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly opt-in to download the Parquet file from Hugging Face if not present.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if the Parquet file already exists locally.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    parquet_path = args.parquet_path

    # Explicit opt-in download handling
    if args.download or args.force_download:
        if args.force_download and parquet_path.is_file():
            parquet_path.unlink()

        url = cfg.get("source_url")
        if not url:
            logger.error("No 'source_url' configured in config file. Cannot download.")
            return 1

        max_size = cfg.get("max_download_size_bytes", MAX_DOWNLOAD_SIZE_BYTES)
        try:
            download_evasionbench_parquet(url, parquet_path, max_size_bytes=max_size)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return 1
    elif not parquet_path.is_file():
        logger.error(
            f"Parquet file not found at: {parquet_path}.\n"
            "To audit a local file, pass --parquet-path <path>.\n"
            "To explicitly download the public approved Parquet file, pass --download."
        )
        return 1

    try:
        result = audit_parquet_dataset(parquet_path, config=cfg)
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        return 1

    # Save JSON output if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(result.to_json(indent=2))
        logger.info(f"Machine-readable audit report written to: {args.output_json}")

    # Print human-readable terminal summary
    print("\n" + "=" * 60)
    print("FINVERITAS EVASIONBENCH DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"File:            {result.filename} ({result.file_size_bytes:,} bytes)")
    print(f"SHA-256:         {result.sha256_hash}")
    print(f"Total Rows:      {result.row_count:,} (Row Groups: {result.row_group_count})")
    print(f"Label Source:    {result.label_provenance}")
    print(f"Pseudo-Labels:   {result.is_model_generated_pseudo_label}")
    print("\nSchema Fields:")
    for col, dtype in result.schema_fields.items():
        print(f"  - {col}: {dtype}")

    print("\nClass Distribution:")
    for cls_name, stats in result.class_distribution.items():
        print(f"  - {cls_name:14s}: {stats['count']:6,d} ({stats['percentage']:5.2f}%)")

    print("\nDuplicate Statistics:")
    for k, v in result.duplicate_statistics.items():
        print(f"  - {k:20s}: {v:,}")

    print("\nQuestion Lengths (characters):")
    for k, v in result.question_length_stats.items():
        print(f"  - {k:8s}: {v}")

    print("\nAnswer Lengths (characters):")
    for k, v in result.answer_length_stats.items():
        print(f"  - {k:8s}: {v}")

    if result.letter_mapping_validation.get("mapping_valid") is not None:
        print(
            f"\nLetter Mapping Valid: {result.letter_mapping_validation['mapping_valid']} "
            f"({result.letter_mapping_validation['mismatches']} mismatches out of "
            f"{result.letter_mapping_validation['total_evaluated']:,})"
        )

    print("\nLeakage Control (Company/Temporal Identifiers):")
    print(f"  - Present: {result.group_temporal_metadata['present_identifiers']}")
    print(f"  - Absent:  {result.group_temporal_metadata['absent_identifiers']}")

    if result.validation_warnings:
        print("\nWarnings:")
        for w in result.validation_warnings:
            print(f"  [WARNING] {w}")

    if result.validation_errors:
        print("\nErrors:")
        for err in result.validation_errors:
            print(f"  [ERROR] {err}")
        return 1

    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
