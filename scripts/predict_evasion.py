#!/usr/bin/env python3
"""
FinVeritas: CLI for Question-Evasion Detection Inference.

Loads the fitted TF-IDF + Logistic Regression baseline pipeline and predicts
one of:
  - direct
  - intermediate
  - fully_evasive

Usage:
  # Train & save model artifact (if not yet trained):
  python scripts/predict_evasion.py --train

  # Inference on an earnings call Q&A pair:
  python scripts/predict_evasion.py --question "..." --answer "..."

  # Inference with JSON output:
  python scripts/predict_evasion.py --question "..." --answer "..." --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from finveritas.nlp_evasion.baselines import (  # noqa: E402
    load_baseline_model,
    predict_evasion,
    train_and_save_baseline_model,
)

logger = logging.getLogger("predict_evasion")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FinVeritas Question-Evasion CLI (TF-IDF Baseline & Neural Transformer)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Analyst question text.",
    )
    parser.add_argument(
        "--answer",
        type=str,
        help="Executive answer text.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tfidf", "transformer"],
        default="tfidf",
        help="Inference engine: 'tfidf' (baseline) or 'transformer' (neural model).",
    )
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Shortcut flag for --model-type transformer.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/initial_baseline/model.joblib"),
        help="Path to the saved TF-IDF model artifact (.joblib).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/transformer_nlp/best_model.pt"),
        help="Path to the saved PyTorch Transformer checkpoint (.pt) when using --transformer.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fit the baseline pipeline on the local train set and save to --model-path.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/raw/evasionbench/evasionbench_17k_eva4b_labels_dedup.parquet"),
        help="Path to the local Parquet dataset used for training.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as a structured JSON object.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Neural inference dispatch
    if args.transformer or args.model_type == "transformer":
        if not (args.question and args.answer):
            logger.error(
                "Both --question and --answer are required for transformer inference.\n"
                "Example: python scripts/predict_evasion.py --transformer "
                '--question "What was Q3 revenue?" --answer "Revenue was $5B."'
            )
            return 1
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from finveritas.nlp_evasion.transformer_classifier import (
                DEFAULT_MAX_LENGTH,
                DEFAULT_MODEL_NAME,
                DISCLAIMER_TEXT,
                ID_TO_LABEL,
                LABEL_TO_ID,
                SUPPORTED_CLASSES,
                predict_evasion_transformer,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer_dir = Path("artifacts/transformer_nlp/tokenizer")
            if tokenizer_dir.is_dir():
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            else:
                tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

            model = AutoModelForSequenceClassification.from_pretrained(
                DEFAULT_MODEL_NAME,
                num_labels=len(SUPPORTED_CLASSES),
                id2label=ID_TO_LABEL,
                label2id=LABEL_TO_ID,
            )
            if args.checkpoint_path.is_file():
                checkpoint = torch.load(args.checkpoint_path, map_location=device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state_dict)
            model.to(device)

            result = predict_evasion_transformer(
                model=model,
                tokenizer=tokenizer,
                question=args.question,
                answer=args.answer,
                device=device,
                max_length=DEFAULT_MAX_LENGTH,
            )

            if args.json:
                print(json.dumps(result, indent=2))
                return 0

            probs = result["uncalibrated_probabilities"]
            print("\n" + "=" * 65)
            print("FINVERITAS QUESTION-EVASION PREDICTION (NEURAL TRANSFORMER)")
            print("=" * 65)
            print(f"Question:   {result['question']}")
            print(f"Answer:     {result['answer']}")
            print(f"Prediction: {result['predicted_label'].upper()}")
            print("\nUncalibrated Class Scores (Pseudo-Label Agreement):")
            for cls_name, prob in probs.items():
                print(f"  - {cls_name:14s}: {prob*100:5.2f}%")
            print(f"\n[DISCLAIMER] {DISCLAIMER_TEXT}")
            print("=" * 65 + "\n")
            return 0
        except Exception as err:
            logger.error(f"Failed to run transformer inference: {err}")
            return 1

    # Handle training flag
    if args.train:
        logger.info(f"Training baseline model from {args.parquet_path}...")
        try:
            train_and_save_baseline_model(
                parquet_path=args.parquet_path,
                model_save_path=args.model_path,
            )
            logger.info(f"Model successfully saved to {args.model_path}")
        except Exception as err:
            logger.error(f"Failed to train and save baseline model: {err}")
            return 1

        if not (args.question and args.answer):
            print(f"[SUCCESS] Model fitted and saved to {args.model_path}.")
            print("To run inference, provide both --question and --answer.")
            return 0

    # Ensure question and answer are provided for inference
    if not (args.question and args.answer):
        logger.error(
            "Both --question and --answer are required for inference.\n"
            "Example: python scripts/predict_evasion.py "
            '--question "What was Q3 revenue?" --answer "Revenue was $5B."'
        )
        return 1

    # Load model and predict
    try:
        pipeline = load_baseline_model(args.model_path)
    except FileNotFoundError as fnf_err:
        logger.error(str(fnf_err))
        return 1
    except Exception as err:
        logger.error(f"Error loading model artifact: {err}")
        return 1

    result = predict_evasion(pipeline, question=args.question, answer=args.answer)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    probs = result["uncalibrated_probabilities"]
    print("\n" + "=" * 60)
    print("FINVERITAS QUESTION-EVASION PREDICTION (CPU-ONLY)")
    print("=" * 60)
    print(f"Question:   {result['question']}")
    print(f"Answer:     {result['answer']}")
    print(f"Prediction: {result['predicted_label'].upper()}")
    print("\nUncalibrated Class Scores (Pseudo-Label Agreement):")
    for cls_name, prob in probs.items():
        print(f"  - {cls_name:14s}: {prob*100:5.2f}%")
    print("\n[DISCLAIMER] These scores reflect linear model decision scores")
    print("trained on Eva-4B-V2 pseudo-labels. They are uncalibrated and")
    print("must not be interpreted as verified real-world confidence estimates.")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
