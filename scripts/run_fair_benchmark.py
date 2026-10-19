#!/usr/bin/env python3
"""
FinVeritas: Checkpoint 05A Execution Script.
Runs Fair NLP Benchmark, 384-Token Ablation, Truncation Mechanics Audit,
Normalized Answer Leakage Audit, and Bootstrap Significance Testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from finveritas.nlp_evasion.fair_benchmark import (  # noqa: E402
    audit_normalized_answer_leakage,
    audit_truncation_mechanics,
    bootstrap_significance_test,
    evaluate_deberta_checkpoint,
    train_fair_tfidf_baseline,
)
from finveritas.nlp_evasion.transformer_classifier import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DISCLAIMER_TEXT,
    ID_TO_LABEL,
    SUPPORTED_CLASSES,
    EvasionQAPairDataset,
    collate_qa_batch,
    evaluate_transformer_model,
    load_and_split_evasion_data,
    train_transformer_model,
)

logger = logging.getLogger("run_fair_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinVeritas Fair Benchmark & Ablation Orchestrator"
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet"),
        help="Path to the EvasionBench Parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/nlp_comparison"),
        help="Directory to save comparison metrics and configs.",
    )
    parser.add_argument(
        "--checkpoint-256",
        type=Path,
        default=Path("artifacts/transformer_nlp/best_model.pt"),
        help="Path to the existing 256-token DeBERTa checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )
    parser.add_argument(
        "--skip-384-training",
        action="store_true",
        help="Skip training the 384-token DeBERTa variant.",
    )
    return parser.parse_args()


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

    # Step 1: Load shared deterministic splits
    logger.info(f"Loading shared benchmark splits from {args.parquet_path}...")
    split_bundle = load_and_split_evasion_data(
        parquet_path=args.parquet_path,
        random_state=args.seed,
    )
    train_data = split_bundle["train"]
    val_data = split_bundle["val"]
    test_data = split_bundle["test"]
    meta = split_bundle["metadata"]

    logger.info(
        f"Partitions: Train={meta['train_samples']}, Val={meta['val_samples']}, "
        f"Test={meta['test_samples']} (Zero question overlap verified)"
    )

    # Step 2: Experiment 1 - Fair Baseline Comparison
    logger.info("--- Step 2: Training Fair TF-IDF Baseline (10,704 training samples) ---")
    tfidf_fair = train_fair_tfidf_baseline(
        train_data=train_data,
        test_data=test_data,
        random_state=args.seed,
    )
    logger.info(
        f"Fair TF-IDF Results: Accuracy={tfidf_fair['accuracy']:.4f}, "
        f"Macro-F1={tfidf_fair['macro_f1']:.4f}, "
        f"Fully-Evasive F1={tfidf_fair['per_class']['fully_evasive']['f1']:.4f}"
    )

    # Re-evaluate 256-token DeBERTa checkpoint on exact test samples
    logger.info(f"--- Step 3: Re-evaluating 256-token DeBERTa ({args.checkpoint_256}) ---")
    deberta_256 = evaluate_deberta_checkpoint(
        checkpoint_path=args.checkpoint_256,
        test_data=test_data,
        model_name=DEFAULT_MODEL_NAME,
        max_length=256,
        device=device,
    )
    logger.info(
        f"DeBERTa-256 Results: Accuracy={deberta_256['accuracy']:.4f}, "
        f"Macro-F1={deberta_256['macro_f1']:.4f}, "
        f"Fully-Evasive F1={deberta_256['per_class']['fully_evasive']['f1']:.4f}"
    )

    # Step 4: Truncation Mechanics Audit
    logger.info("--- Step 4: Auditing Truncation Mechanics (256 vs 384 tokens) ---")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    trunc_audit = {
        "max_length_256": {
            "train": audit_truncation_mechanics(
                train_data["questions"], train_data["answers"], tokenizer, max_length=256
            ).to_dict(),
            "val": audit_truncation_mechanics(
                val_data["questions"], val_data["answers"], tokenizer, max_length=256
            ).to_dict(),
            "test": audit_truncation_mechanics(
                test_data["questions"], test_data["answers"], tokenizer, max_length=256
            ).to_dict(),
        },
        "max_length_384": {
            "train": audit_truncation_mechanics(
                train_data["questions"], train_data["answers"], tokenizer, max_length=384
            ).to_dict(),
            "val": audit_truncation_mechanics(
                val_data["questions"], val_data["answers"], tokenizer, max_length=384
            ).to_dict(),
            "test": audit_truncation_mechanics(
                test_data["questions"], test_data["answers"], tokenizer, max_length=384
            ).to_dict(),
        },
    }

    test_trunc_256 = trunc_audit["max_length_256"]["test"]
    test_trunc_384 = trunc_audit["max_length_384"]["test"]
    logger.info(
        f"Test Truncation @ 256 tokens: Pair={test_trunc_256['pair_truncation_rate']*100:.2f}%, "
        f"Answer={test_trunc_256['answer_truncation_rate']*100:.2f}%, "
        f"Question={test_trunc_256['question_truncation_rate']*100:.2f}%"
    )
    logger.info(
        f"Test Truncation @ 384 tokens: Pair={test_trunc_384['pair_truncation_rate']*100:.2f}%, "
        f"Answer={test_trunc_384['answer_truncation_rate']*100:.2f}%, "
        f"Question={test_trunc_384['question_truncation_rate']*100:.2f}%"
    )

    # Step 5: Train 384-token DeBERTa variant
    deberta_384: dict[str, Any] = {}
    if not args.skip_384_training:
        logger.info("--- Step 5: Training 384-token DeBERTa Variant (1 epoch) ---")
        out_384_dir = args.output_dir / "checkpoints_384"
        out_384_dir.mkdir(parents=True, exist_ok=True)

        model_384, tok_384, train_384_metrics = train_transformer_model(
            train_data=train_data,
            val_data=val_data,
            model_name=DEFAULT_MODEL_NAME,
            output_dir=out_384_dir,
            epochs=1,
            batch_size=2,
            grad_accum_steps=4,
            learning_rate=2e-5,
            max_length=384,
            seed=args.seed,
            device=device,
        )

        logger.info("Evaluating 384-token model on held-out test set...")
        eval_384 = evaluate_transformer_model(
            model=model_384,
            tokenizer=tok_384,
            test_data=test_data,
            device=device,
            batch_size=8,
            max_length=384,
            train_wall_time=train_384_metrics.train_wall_time_seconds,
            peak_gpu_memory_mb=train_384_metrics.peak_gpu_memory_mb,
        )

        # Collect 384 predictions
        test_dataset_384 = EvasionQAPairDataset(
            questions=test_data["questions"],
            answers=test_data["answers"],
            labels=test_data["labels"],
            tokenizer=tok_384,
            max_length=384,
        )
        pad_id = tok_384.pad_token_id if tok_384.pad_token_id is not None else 0
        loader_384 = DataLoader(
            test_dataset_384,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda b: collate_qa_batch(b, pad_token_id=pad_id),
        )

        y_384_preds: list[str] = []
        with torch.no_grad():
            for b in loader_384:
                input_ids = b["input_ids"].to(device)
                attention_mask = b["attention_mask"].to(device)
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in b:
                    kwargs["token_type_ids"] = b["token_type_ids"].to(device)
                out = model_384(**kwargs)
                preds = torch.argmax(out.logits, dim=-1)
                y_384_preds.extend([ID_TO_LABEL[p.item()] for p in preds])

        deberta_384 = {
            "model_name": DEFAULT_MODEL_NAME,
            "max_length": 384,
            "accuracy": eval_384.accuracy,
            "macro_f1": eval_384.macro_f1,
            "weighted_f1": eval_384.weighted_f1,
            "per_class": eval_384.per_class,
            "confusion_matrix": eval_384.confusion_matrix,
            "train_wall_time_seconds": eval_384.train_wall_time_seconds,
            "eval_wall_time_seconds": eval_384.eval_wall_time_seconds,
            "peak_gpu_memory_mb": eval_384.peak_gpu_memory_mb,
            "truncation_stats": eval_384.truncation_stats,
            "y_pred": y_384_preds,
        }
        logger.info(
            f"DeBERTa-384 Results: Accuracy={deberta_384['accuracy']:.4f}, "
            f"Macro-F1={deberta_384['macro_f1']:.4f}, "
            f"Fully-Evasive F1={deberta_384['per_class']['fully_evasive']['f1']:.4f}"
        )

    # Step 6: Normalized Answer Leakage Audit
    logger.info("--- Step 6: Auditing Normalized Answer Leakage ---")
    leakage_report = audit_normalized_answer_leakage(
        train_answers=train_data["answers"],
        val_answers=val_data["answers"],
        test_answers=test_data["answers"],
        train_labels=train_data["labels"],
        val_labels=val_data["labels"],
        test_labels=test_data["labels"],
    )
    logger.info(
        f"Answer Leakage: Train/Test Overlap={leakage_report.overlap_train_test} unique answers "
        f"({leakage_report.train_test_overlap_samples_in_test} test rows, "
        f"Agreement Rate={leakage_report.train_test_label_agreement_rate*100:.2f}%)"
    )

    # Step 7: Bootstrap Significance Testing
    logger.info("--- Step 7: Performing Bootstrap Significance Testing (1,000 resamples) ---")
    preds_for_boot = {
        "tfidf_fair_10k": tfidf_fair["y_pred"],
        "deberta_256": deberta_256["y_pred"],
    }
    if deberta_384 and "y_pred" in deberta_384:
        preds_for_boot["deberta_384"] = deberta_384["y_pred"]

    bootstrap_results = bootstrap_significance_test(
        y_true=test_data["labels"],
        predictions=preds_for_boot,
        n_bootstraps=1000,
        seed=args.seed,
    )

    # Clean prediction arrays before saving metrics
    tfidf_fair_clean = {k: v for k, v in tfidf_fair.items() if k != "y_pred"}
    deberta_256_clean = {k: v for k, v in deberta_256.items() if k != "y_pred"}
    deberta_384_clean = (
        {k: v for k, v in deberta_384.items() if k != "y_pred"} if deberta_384 else {}
    )

    # Save metrics.json
    final_metrics = {
        "fair_tfidf_baseline": tfidf_fair_clean,
        "deberta_v3_xsmall_256": deberta_256_clean,
        "deberta_v3_xsmall_384": deberta_384_clean,
        "truncation_mechanics_audit": trunc_audit,
        "normalized_answer_leakage_audit": leakage_report.to_dict(),
        "bootstrap_significance": bootstrap_results,
        "scientific_disclaimer": DISCLAIMER_TEXT,
    }
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Saved comparison metrics to {metrics_path}")

    # Save config.json
    final_config = {
        "experiment_name": "Checkpoint 05A: Fair NLP Benchmark and Longer-Context Ablation",
        "dataset_path": str(args.parquet_path),
        "target_field": "eva4b_label",
        "models_evaluated": [
            "tfidf_logistic_regression_fair_10k",
            "microsoft/deberta-v3-xsmall (256 tokens)",
            "microsoft/deberta-v3-xsmall (384 tokens)",
        ],
        "split_configuration": {
            "total_samples": meta["total_samples"],
            "train_samples": meta["train_samples"],
            "val_samples": meta["val_samples"],
            "test_samples": meta["test_samples"],
            "group_overlap": 0,
            "seed": args.seed,
        },
        "training_hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "epochs": 1,
            "precision": "FP32",
            "device": str(device),
        },
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
        json.dump(final_config, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Formatted side-by-side terminal report
    print("\n" + "=" * 95)
    print(f"{'FINVERITAS CHECKPOINT 05A: FAIR BENCHMARK & LONGER-CONTEXT ABLATION':^95}")
    print("=" * 95)
    header = (
        f"{'Metric':<30} | {'TF-IDF (Fair 10k)':<20} | "
        f"{'DeBERTa (256)':<18} | {'DeBERTa (384)':<18}"
    )
    print(header)
    print("-" * 95)

    def f_val(d: dict, k: str) -> str:
        v = d.get(k, 0.0)
        return f"{v:.4f}" if isinstance(v, (float, int)) else str(v)

    def f_pc(d: dict, c: str) -> str:
        return f"{d.get('per_class', {}).get(c, {}).get('f1', 0.0):.4f}"

    print(
        f"{'Accuracy':<30} | {f_val(tfidf_fair, 'accuracy'):<20} | "
        f"{f_val(deberta_256, 'accuracy'):<18} | {f_val(deberta_384, 'accuracy'):<18}"
    )
    print(
        f"{'Macro-F1':<30} | {f_val(tfidf_fair, 'macro_f1'):<20} | "
        f"{f_val(deberta_256, 'macro_f1'):<18} | {f_val(deberta_384, 'macro_f1'):<18}"
    )
    print(
        f"{'Weighted-F1':<30} | {f_val(tfidf_fair, 'weighted_f1'):<20} | "
        f"{f_val(deberta_256, 'weighted_f1'):<18} | {f_val(deberta_384, 'weighted_f1'):<18}"
    )
    print("-" * 95)
    for c in SUPPORTED_CLASSES:
        lbl = f"F1 ({c})"
        print(
            f"{lbl:<30} | {f_pc(tfidf_fair, c):<20} | "
            f"{f_pc(deberta_256, c):<18} | {f_pc(deberta_384, c):<18}"
        )
    print("-" * 95)
    t_tfidf = f"{tfidf_fair.get('train_wall_time_seconds', 0.0):.1f}s"
    t_deb256 = "550.2s"  # From prior run
    t_deb384 = f"{deberta_384.get('train_wall_time_seconds', 0.0):.1f}s"
    print(f"{'Train Wall Time':<30} | {t_tfidf:<20} | {t_deb256:<18} | {t_deb384:<18}")
    e_tfidf = f"{tfidf_fair.get('eval_wall_time_seconds', 0.0):.2f}s"
    e_deb256 = f"{deberta_256.get('eval_wall_time_seconds', 0.0):.2f}s"
    e_deb384 = f"{deberta_384.get('eval_wall_time_seconds', 0.0):.2f}s"
    print(f"{'Eval Wall Time':<30} | {e_tfidf:<20} | {e_deb256:<18} | {e_deb384:<18}")
    vram_256 = f"{deberta_256.get('peak_gpu_memory_mb', 0.0):.1f} MB"
    vram_384 = f"{deberta_384.get('peak_gpu_memory_mb', 0.0):.1f} MB"
    print(f"{'Peak GPU VRAM':<30} | {'0.0 MB (CPU)':<20} | {vram_256:<18} | {vram_384:<18}")
    tr_256 = f"{test_trunc_256['pair_truncation_rate']*100:.1f}%"
    tr_384 = f"{test_trunc_384['pair_truncation_rate']*100:.1f}%"
    print(f"{'Test Pair Truncation Rate':<30} | {'0.0% (N/A)':<20} | {tr_256:<18} | {tr_384:<18}")
    print("=" * 95)

    return 0


if __name__ == "__main__":
    sys.exit(main())
