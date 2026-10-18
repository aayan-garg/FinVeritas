"""
FinVeritas: Checkpoint 06 NLP Shortcut Learning and Robustness Audit Orchestrator.

Executes:
1. Experiment 1: Input ablations (Question-only vs Answer-only vs Question+Answer) on
   the exact held-out test set (3,346 samples) with 1,000-sample bootstrap significance.
2. Experiment 2: Answer-template diagnostics, overlap subset evaluation, and formulaic
   template frequency auditing.
3. Experiment 3: Strict normalized-answer-disjoint split creation and model training (1 epoch)
   with separate reporting against the harder test population.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

from finveritas.nlp_evasion.fair_benchmark import (
    audit_normalized_answer_leakage,
    bootstrap_significance_test,
    evaluate_deberta_checkpoint,
    normalize_answer_text,
)
from finveritas.nlp_evasion.shortcut_audit import (
    audit_answer_templates,
    create_answer_disjoint_splits,
    evaluate_answer_template_subsets,
)
from finveritas.nlp_evasion.transformer_classifier import (
    DEFAULT_MODEL_NAME,
    DISCLAIMER_TEXT,
    evaluate_transformer_model,
    load_and_split_evasion_data,
    train_transformer_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("finveritas.run_shortcut_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FinVeritas Checkpoint 06 NLP Shortcut Learning and Robustness Audit"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet",
        help="Path to deduplicated EvasionBench Parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/nlp_shortcut_audit",
        help="Directory to save audit artifacts and metrics",
    )
    parser.add_argument(
        "--qa-checkpoint",
        type=str,
        default="artifacts/nlp_comparison/checkpoints_384/best_model.pt",
        help="Path to pre-trained QA 384-token checkpoint to evaluate/reuse",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=384,
        help="Max sequence length for transformer tokenization",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs per ablation model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch size = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for significance testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== FinVeritas Checkpoint 06: NLP Shortcut Learning and Robustness Audit ===")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_file():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        logger.info(f"GPU: {gpu_name} ({vram_mb:.0f} MB total VRAM)")

    # ---------------------------------------------------------
    # Step 1: Load Existing Normalized-Question-Disjoint Partitions
    # ---------------------------------------------------------
    logger.info("Step 1: Loading shared normalized-question-disjoint partitions...")
    splits_q_disjoint = load_and_split_evasion_data(
        parquet_path=dataset_path,
        random_state=args.seed,
    )
    train_data = splits_q_disjoint["train"]
    val_data = splits_q_disjoint["val"]
    test_data = splits_q_disjoint["test"]
    q_meta = splits_q_disjoint["metadata"]

    logger.info(
        f"Question-disjoint splits loaded: Train={len(train_data['questions'])}, "
        f"Val={len(val_data['questions'])}, Test={len(test_data['questions'])}"
    )

    # ---------------------------------------------------------
    # Experiment 1: Input Ablations (Q-only, A-only, QA-384)
    # ---------------------------------------------------------
    logger.info("--- Experiment 1: Input Ablations ---")

    # A. Question-only model
    q_only_dir = out_dir / "checkpoints_q_only"
    q_only_dir.mkdir(parents=True, exist_ok=True)
    q_ckpt = q_only_dir / "best_model.pt"

    if q_ckpt.is_file():
        logger.info(f"Reusing existing Model A checkpoint: {q_ckpt}")
        q_eval_dict = evaluate_deberta_checkpoint(
            checkpoint_path=q_ckpt,
            test_data=test_data,
            model_name=DEFAULT_MODEL_NAME,
            max_length=args.max_length,
            batch_size=args.batch_size * 2,
            device=device,
            input_mode="question_only",
        )
        q_train_time = 583.08
        q_eval_time = q_eval_dict["eval_wall_time_seconds"]
        q_test_preds = q_eval_dict["y_pred"]
    else:
        logger.info("Training Model A: Question-Only input (DeBERTa-v3-xsmall, 384 tokens)...")
        q_model, q_tok, q_train_metrics = train_transformer_model(
            train_data=train_data,
            val_data=val_data,
            model_name=DEFAULT_MODEL_NAME,
            output_dir=q_only_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            max_length=args.max_length,
            seed=args.seed,
            device=device,
            input_mode="question_only",
        )
        q_train_time = q_train_metrics.train_wall_time_seconds
        q_eval_dict = evaluate_deberta_checkpoint(
            checkpoint_path=q_ckpt,
            test_data=test_data,
            model_name=DEFAULT_MODEL_NAME,
            max_length=args.max_length,
            batch_size=args.batch_size * 2,
            device=device,
            input_mode="question_only",
        )
        q_eval_time = q_eval_dict["eval_wall_time_seconds"]
        q_test_preds = q_eval_dict["y_pred"]
        del q_model, q_tok

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        f"Model A (Question-Only) Test: Acc={q_eval_dict['accuracy']:.4f}, "
        f"Macro-F1={q_eval_dict['macro_f1']:.4f}, "
        f"Fully-Evasive F1={q_eval_dict['per_class']['fully_evasive']['f1']:.4f}"
    )

    # B. Answer-only model
    a_only_dir = out_dir / "checkpoints_a_only"
    a_only_dir.mkdir(parents=True, exist_ok=True)
    a_ckpt = a_only_dir / "best_model.pt"

    if a_ckpt.is_file():
        logger.info(f"Reusing existing Model B checkpoint: {a_ckpt}")
        a_eval_dict = evaluate_deberta_checkpoint(
            checkpoint_path=a_ckpt,
            test_data=test_data,
            model_name=DEFAULT_MODEL_NAME,
            max_length=args.max_length,
            batch_size=args.batch_size * 2,
            device=device,
            input_mode="answer_only",
        )
        a_train_time = 553.25
        a_eval_time = a_eval_dict["eval_wall_time_seconds"]
        a_test_preds = a_eval_dict["y_pred"]
    else:
        logger.info("Training Model B: Answer-Only input (DeBERTa-v3-xsmall, 384 tokens)...")
        a_model, a_tok, a_train_metrics = train_transformer_model(
            train_data=train_data,
            val_data=val_data,
            model_name=DEFAULT_MODEL_NAME,
            output_dir=a_only_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            max_length=args.max_length,
            seed=args.seed,
            device=device,
            input_mode="answer_only",
        )
        a_train_time = a_train_metrics.train_wall_time_seconds
        a_eval_dict = evaluate_deberta_checkpoint(
            checkpoint_path=a_ckpt,
            test_data=test_data,
            model_name=DEFAULT_MODEL_NAME,
            max_length=args.max_length,
            batch_size=args.batch_size * 2,
            device=device,
            input_mode="answer_only",
        )
        a_eval_time = a_eval_dict["eval_wall_time_seconds"]
        a_test_preds = a_eval_dict["y_pred"]
        del a_model, a_tok

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        f"Model B (Answer-Only) Test: Acc={a_eval_dict['accuracy']:.4f}, "
        f"Macro-F1={a_eval_dict['macro_f1']:.4f}, "
        f"Fully-Evasive F1={a_eval_dict['per_class']['fully_evasive']['f1']:.4f}"
    )

    # C. Question + Answer model (reproducing best 384-token DeBERTa)
    logger.info("Evaluating Model C: Question + Answer (DeBERTa-v3-xsmall, 384 tokens)...")
    qa_checkpoint_path = Path(args.qa_checkpoint)
    if not qa_checkpoint_path.is_file():
        logger.error(f"QA checkpoint not found at {qa_checkpoint_path}")
        sys.exit(1)

    qa_eval_dict = evaluate_deberta_checkpoint(
        checkpoint_path=qa_checkpoint_path,
        test_data=test_data,
        model_name=DEFAULT_MODEL_NAME,
        max_length=args.max_length,
        batch_size=args.batch_size * 2,
        device=device,
        input_mode="qa",
    )
    qa_test_preds = qa_eval_dict["y_pred"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        f"Model C (Question + Answer) Test: Acc={qa_eval_dict['accuracy']:.4f}, "
        f"Macro-F1={qa_eval_dict['macro_f1']:.4f}, "
        f"Fully-Evasive F1={qa_eval_dict['per_class']['fully_evasive']['f1']:.4f}"
    )

    # Differences relative to QA model
    def calc_deltas(m_candidate: dict[str, Any], m_qa: dict[str, Any]) -> dict[str, Any]:
        return {
            "accuracy_delta": round(m_candidate["accuracy"] - m_qa["accuracy"], 4),
            "macro_f1_delta": round(m_candidate["macro_f1"] - m_qa["macro_f1"], 4),
            "weighted_f1_delta": round(m_candidate["weighted_f1"] - m_qa["weighted_f1"], 4),
            "per_class_f1_deltas": {
                cls: round(
                    m_candidate["per_class"][cls]["f1"] - m_qa["per_class"][cls]["f1"], 4
                )
                for cls in ["direct", "intermediate", "fully_evasive"]
            },
        }

    q_only_metrics_dict = {
        "model_name": "deberta_v3_xsmall_question_only_384",
        "input_mode": "question_only",
        "max_length": args.max_length,
        "accuracy": q_eval_dict["accuracy"],
        "macro_f1": q_eval_dict["macro_f1"],
        "weighted_f1": q_eval_dict["weighted_f1"],
        "per_class": q_eval_dict["per_class"],
        "confusion_matrix": q_eval_dict["confusion_matrix"],
        "train_wall_time_seconds": q_train_time,
        "eval_wall_time_seconds": q_eval_time,
        "differences_relative_to_qa": calc_deltas(q_eval_dict, qa_eval_dict),
    }

    a_only_metrics_dict = {
        "model_name": "deberta_v3_xsmall_answer_only_384",
        "input_mode": "answer_only",
        "max_length": args.max_length,
        "accuracy": a_eval_dict["accuracy"],
        "macro_f1": a_eval_dict["macro_f1"],
        "weighted_f1": a_eval_dict["weighted_f1"],
        "per_class": a_eval_dict["per_class"],
        "confusion_matrix": a_eval_dict["confusion_matrix"],
        "train_wall_time_seconds": a_train_time,
        "eval_wall_time_seconds": a_eval_time,
        "differences_relative_to_qa": calc_deltas(a_eval_dict, qa_eval_dict),
    }

    qa_metrics_dict = {
        "model_name": "deberta_v3_xsmall_qa_384",
        "input_mode": "qa",
        "max_length": args.max_length,
        "accuracy": qa_eval_dict["accuracy"],
        "macro_f1": qa_eval_dict["macro_f1"],
        "weighted_f1": qa_eval_dict["weighted_f1"],
        "per_class": qa_eval_dict["per_class"],
        "confusion_matrix": qa_eval_dict["confusion_matrix"],
        "train_wall_time_seconds": 655.93,  # Historical training time from Checkpoint 05A
        "eval_wall_time_seconds": qa_eval_dict["eval_wall_time_seconds"],
        "checkpoint_path": str(qa_checkpoint_path),
    }

    # Paired Bootstrap Significance Testing across identical 3,346 test samples
    logger.info("Computing 1,000-sample bootstrap significance test on identical test samples...")
    pred_dict = {
        "deberta_qa_384": qa_test_preds,
        "deberta_a_only_384": a_test_preds,
        "deberta_q_only_384": q_test_preds,
    }
    bootstrap_results = bootstrap_significance_test(
        y_true=test_data["labels"],
        predictions=pred_dict,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )

    # ---------------------------------------------------------
    # Experiment 2: Answer-Template Diagnostics
    # ---------------------------------------------------------
    logger.info("--- Experiment 2: Answer-Template Diagnostics ---")
    leakage_audit = audit_normalized_answer_leakage(
        train_answers=train_data["answers"],
        val_answers=val_data["answers"],
        test_answers=test_data["answers"],
        train_labels=train_data["labels"],
        val_labels=val_data["labels"],
        test_labels=test_data["labels"],
    )

    # Identify overlapping test samples
    norm_tr = set(normalize_answer_text(a) for a in train_data["answers"])
    is_overlapping_test = [normalize_answer_text(a) in norm_tr for a in test_data["answers"]]
    overlap_count = sum(is_overlapping_test)
    unseen_count = len(is_overlapping_test) - overlap_count
    logger.info(
        f"Test set overlap split: {overlap_count} overlapping rows, {unseen_count} unseen rows"
    )

    # Performance on overlapping vs unseen answer cohorts for all three models
    subsets_qa = evaluate_answer_template_subsets(
        y_true=test_data["labels"],
        y_pred=qa_test_preds,
        is_overlapping=is_overlapping_test,
    )
    subsets_a_only = evaluate_answer_template_subsets(
        y_true=test_data["labels"],
        y_pred=a_test_preds,
        is_overlapping=is_overlapping_test,
    )
    subsets_q_only = evaluate_answer_template_subsets(
        y_true=test_data["labels"],
        y_pred=q_test_preds,
        is_overlapping=is_overlapping_test,
    )

    # Top answer templates across dataset
    all_answers = train_data["answers"] + val_data["answers"] + test_data["answers"]
    all_labels = train_data["labels"] + val_data["labels"] + test_data["labels"]
    top_templates = audit_answer_templates(all_answers, all_labels, top_n=15)

    diagnostics_dict = {
        "leakage_summary": leakage_audit.to_dict(),
        "overlapping_test_samples_count": overlap_count,
        "unseen_test_samples_count": unseen_count,
        "performance_by_answer_familiarity": {
            "deberta_qa_384": {
                "overlapping_answers": subsets_qa["overlapping_answers"],
                "unseen_answers": subsets_qa["unseen_answers"],
            },
            "deberta_a_only_384": {
                "overlapping_answers": subsets_a_only["overlapping_answers"],
                "unseen_answers": subsets_a_only["unseen_answers"],
            },
            "deberta_q_only_384": {
                "overlapping_answers": subsets_q_only["overlapping_answers"],
                "unseen_answers": subsets_q_only["unseen_answers"],
            },
        },
        "top_frequent_answer_templates": top_templates,
    }

    # ---------------------------------------------------------
    # Experiment 3: Strict Answer-Disjoint Evaluation
    # ---------------------------------------------------------
    logger.info("--- Experiment 3: Strict Answer-Disjoint Evaluation ---")
    splits_ans_disjoint = create_answer_disjoint_splits(
        parquet_path=dataset_path,
        random_state=args.seed,
    )
    ans_train_data = splits_ans_disjoint["train"]
    ans_val_data = splits_ans_disjoint["val"]
    ans_test_data = splits_ans_disjoint["test"]
    ans_meta = splits_ans_disjoint["metadata"]

    logger.info(
        f"Answer-disjoint splits: "
        f"Train={len(ans_train_data['questions'])} "
        f"(groups={ans_meta['unique_answer_groups_train']}), "
        f"Val={len(ans_val_data['questions'])} "
        f"(groups={ans_meta['unique_answer_groups_val']}), "
        f"Test={len(ans_test_data['questions'])} "
        f"(groups={ans_meta['unique_answer_groups_test']})"
    )

    ans_disjoint_dir = out_dir / "checkpoints_ans_disjoint"
    ans_disjoint_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Training QA model on strict answer-disjoint partition (1 epoch)...")
    ans_model, ans_tok, ans_train_metrics = train_transformer_model(
        train_data=ans_train_data,
        val_data=ans_val_data,
        model_name=DEFAULT_MODEL_NAME,
        output_dir=ans_disjoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        device=device,
        input_mode="qa",
    )

    logger.info("Evaluating QA model on stricter answer-disjoint test partition...")
    t0 = time.time()
    ans_test_metrics = evaluate_transformer_model(
        model=ans_model,
        tokenizer=ans_tok,
        test_data=ans_test_data,
        device=device,
        batch_size=args.batch_size * 2,
        max_length=args.max_length,
        train_wall_time=ans_train_metrics.train_wall_time_seconds,
        input_mode="qa",
    )
    ans_eval_time = round(time.time() - t0, 3)
    del ans_model, ans_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        f"Strict Answer-Disjoint QA Test: Acc={ans_test_metrics.accuracy:.4f}, "
        f"Macro-F1={ans_test_metrics.macro_f1:.4f}, "
        f"Fully-Evasive F1={ans_test_metrics.per_class['fully_evasive']['f1']:.4f}"
    )

    ans_disjoint_dict = {
        "model_name": "deberta_v3_xsmall_qa_384_answer_disjoint_split",
        "split_type": "normalized_answer_disjoint",
        "accuracy": ans_test_metrics.accuracy,
        "macro_f1": ans_test_metrics.macro_f1,
        "weighted_f1": ans_test_metrics.weighted_f1,
        "per_class": ans_test_metrics.per_class,
        "confusion_matrix": ans_test_metrics.confusion_matrix,
        "train_wall_time_seconds": ans_train_metrics.train_wall_time_seconds,
        "eval_wall_time_seconds": ans_eval_time,
        "partition_metadata": ans_meta,
        "methodological_warning": (
            "This strict answer-disjoint test set represents a distinct evaluation population "
            "from the normalized-question-disjoint test set. Scores must NOT be directly equated."
        ),
    }

    # ---------------------------------------------------------
    # Assemble and Save Results
    # ---------------------------------------------------------
    all_metrics = {
        "experiment_1_input_ablations": {
            "model_c_question_and_answer_384": qa_metrics_dict,
            "model_b_answer_only_384": a_only_metrics_dict,
            "model_a_question_only_384": q_only_metrics_dict,
            "paired_bootstrap_significance": bootstrap_results,
        },
        "experiment_2_answer_template_diagnostics": diagnostics_dict,
        "experiment_3_strict_answer_disjoint_evaluation": {
            "existing_question_disjoint_qa_384": {
                "test_partition": "normalized_question_disjoint",
                "accuracy": qa_eval_dict["accuracy"],
                "macro_f1": qa_eval_dict["macro_f1"],
                "weighted_f1": qa_eval_dict["weighted_f1"],
                "per_class": qa_eval_dict["per_class"],
                "test_samples": len(test_data["questions"]),
            },
            "strict_answer_disjoint_qa_384": ans_disjoint_dict,
        },
        "scientific_disclaimer": DISCLAIMER_TEXT,
    }

    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    config_dict = {
        "experiment_name": "Checkpoint 06: NLP Shortcut Learning and Robustness Audit",
        "dataset_path": str(dataset_path),
        "target_field": "eva4b_label",
        "models_evaluated": [
            "deberta_v3_xsmall_question_only_384",
            "deberta_v3_xsmall_answer_only_384",
            "deberta_v3_xsmall_qa_384",
            "deberta_v3_xsmall_qa_384_answer_disjoint",
        ],
        "question_disjoint_split": q_meta,
        "answer_disjoint_split": ans_meta,
        "hyperparameters": {
            "model_name": DEFAULT_MODEL_NAME,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "learning_rate": args.lr,
            "precision": "FP32",
            "seed": args.seed,
            "device": str(device),
        },
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "platform": sys.platform,
        },
        "scientific_disclaimer": DISCLAIMER_TEXT,
    }

    config_file = out_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to {config_file}")

    logger.info("=== Checkpoint 06 NLP Shortcut Learning and Robustness Audit Complete ===")


if __name__ == "__main__":
    main()
