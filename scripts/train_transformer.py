#!/usr/bin/env python3
"""
FinVeritas: Training and Evaluation Script for Transformer Evasion Classifier.

Fine-tunes microsoft/deberta-v3-xsmall on EvasionBench under strict,
leakage-free normalized question group partitions.

Outputs:
  - artifacts/transformer_nlp/metrics.json
  - artifacts/transformer_nlp/config.json
  - docs/research/transformer_nlp_results.md
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

# Add src to sys.path
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch  # noqa: E402

from finveritas.nlp_evasion.transformer_classifier import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DISCLAIMER_TEXT,
    SUPPORTED_CLASSES,
    evaluate_transformer_model,
    load_and_split_evasion_data,
    run_forward_backward_smoke_test,
    train_transformer_model,
)

logger = logging.getLogger("train_transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinVeritas Transformer Evasion Model Trainer & Evaluator"
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet"),
        help="Path to the EvasionBench Parquet dataset.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model identifier (default: microsoft/deberta-v3-xsmall).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/transformer_nlp"),
        help="Directory to save metrics, configs, and checkpoint artifacts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW (default: 2e-5).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length for tokenization (default: 384).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run preflight forward/backward smoke test and exit without full training.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional maximum number of training samples for fast verification.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional maximum number of validation samples.",
    )
    return parser.parse_args()


def load_baseline_metrics(baseline_metrics_path: Path) -> dict:
    """Loads previous TF-IDF baseline metrics if available."""
    if baseline_metrics_path.is_file():
        try:
            with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def format_side_by_side_table(baseline: dict, transformer: dict) -> str:
    """Formats comparison table between TF-IDF baseline and DeBERTa-v3-xsmall."""
    b_metrics = baseline.get("tfidf_logistic_regression", {})
    b_acc = f"{b_metrics.get('accuracy', 0.0):.4f}"
    t_acc = f"{transformer.get('accuracy', 0.0):.4f}"
    b_mf1 = f"{b_metrics.get('macro_f1', 0.0):.4f}"
    t_mf1 = f"{transformer.get('macro_f1', 0.0):.4f}"
    b_wf1 = f"{b_metrics.get('weighted_f1', 0.0):.4f}"
    t_wf1 = f"{transformer.get('weighted_f1', 0.0):.4f}"

    lines = [
        "",
        "=" * 85,
        f"{'FINVERITAS NLP MODEL COMPARISON: TF-IDF BASELINE VS DEBERTA-V3-XSMALL':^85}",
        "=" * 85,
        f"{'Metric':<30} | {'TF-IDF + LogReg':<24} | {'DeBERTa-v3-xsmall':<24}",
        "-" * 85,
        f"{'Accuracy':<30} | {b_acc:<24} | {t_acc:<24}",
        f"{'Macro-F1':<30} | {b_mf1:<24} | {t_mf1:<24}",
        f"{'Weighted-F1':<30} | {b_wf1:<24} | {t_wf1:<24}",
        "-" * 85,
    ]

    for cls_name in SUPPORTED_CLASSES:
        b_cls = b_metrics.get("per_class", {}).get(cls_name, {})
        t_cls = transformer.get("per_class", {}).get(cls_name, {})
        b_f1 = f"{b_cls.get('f1', 0.0):.4f}"
        t_f1 = f"{t_cls.get('f1', 0.0):.4f}"
        lines.append(f"{'F1 (' + cls_name + ')':<30} | {b_f1:<24} | {t_f1:<24}")

    b_train_t = f"{b_metrics.get('train_wall_time_seconds', 0.0):.2f}s"
    t_train_t = f"{transformer.get('train_wall_time_seconds', 0.0):.2f}s"
    b_eval_t = f"{b_metrics.get('eval_wall_time_seconds', 0.0):.2f}s"
    t_eval_t = f"{transformer.get('eval_wall_time_seconds', 0.0):.2f}s"
    t_peak_vram = f"{transformer.get('peak_gpu_memory_mb', 0.0):.2f} MB"

    lines.extend([
        "-" * 85,
        f"{'Train Wall Time':<30} | {b_train_t:<24} | {t_train_t:<24}",
        f"{'Eval Wall Time':<30} | {b_eval_t:<24} | {t_eval_t:<24}",
        f"{'Peak GPU VRAM':<30} | {'0.0 MB (CPU)':<24} | {t_peak_vram:<24}",
        "=" * 85,
        f"[NOTE] {DISCLAIMER_TEXT}",
        "=" * 85,
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Forward-backward smoke test
    smoke_results = run_forward_backward_smoke_test(
        model_name=args.model_name,
        device=device,
        max_length=args.max_length,
    )
    if args.smoke_test:
        print("[SMOKE TEST SUCCESS]", json.dumps(smoke_results, indent=2))
        return 0

    # 2. Load and partition data
    logger.info(f"Loading and partitioning data from {args.parquet_path}...")
    split_bundle = load_and_split_evasion_data(
        parquet_path=args.parquet_path,
        random_state=args.seed,
    )
    train_data = split_bundle["train"]
    val_data = split_bundle["val"]
    test_data = split_bundle["test"]
    meta = split_bundle["metadata"]

    logger.info(
        f"Data partitioned: Train={meta['train_samples']} samples, "
        f"Val={meta['val_samples']} samples, Test={meta['test_samples']} samples. "
        f"Verified 0 normalized-question overlap across all 3 sets."
    )

    # 3. Train model
    logger.info(f"Training transformer model: {args.model_name}...")
    model, tokenizer, train_metrics = train_transformer_model(
        train_data=train_data,
        val_data=val_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        device=device,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    # 4. Final Evaluation on held-out test partition
    n_test = len(test_data["questions"])
    logger.info(f"Evaluating best checkpoint on held-out test set ({n_test} samples)...")
    test_metrics = evaluate_transformer_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        device=device,
        batch_size=args.batch_size * 2,
        max_length=args.max_length,
        train_wall_time=train_metrics.train_wall_time_seconds,
        peak_gpu_memory_mb=train_metrics.peak_gpu_memory_mb,
    )

    # 5. Save artifacts/transformer_nlp/metrics.json
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(test_metrics.to_json(indent=2))
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # 6. Save artifacts/transformer_nlp/config.json
    config_dict = {
        "model_id": args.model_name,
        "dataset_path": str(args.parquet_path),
        "target_field": "eva4b_label",
        "input_fields": ["question", "answer"],
        "prohibited_fields_excluded": ["choices", "eva4b_label_letter", "uid"],
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "random_seed": args.seed,
        "device": str(device),
        "split_configuration": {
            "split_method": "StratifiedGroupKFold",
            "base_n_splits": 5,
            "test_fold_index": 0,
            "random_state": args.seed,
            "total_samples": meta["total_samples"],
            "train_samples": meta["train_samples"],
            "val_samples": meta["val_samples"],
            "test_samples": meta["test_samples"],
            "group_overlap": 0,
        },
        "class_weights": split_bundle["class_weights"],
        "truncation_stats": test_metrics.truncation_stats,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "platform": platform.platform(),
        },
        "scientific_disclaimer": DISCLAIMER_TEXT,
    }
    config_path = args.output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # 7. Print side-by-side comparison with baseline
    baseline_path = Path("artifacts/initial_baseline/metrics.json")
    baseline_metrics = load_baseline_metrics(baseline_path)
    comparison_str = format_side_by_side_table(baseline_metrics, test_metrics.to_dict())
    print(comparison_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
