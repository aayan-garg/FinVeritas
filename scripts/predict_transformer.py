#!/usr/bin/env python3
"""
FinVeritas: CLI for Question-Evasion Detection Inference using Transformer Model.

Loads fine-tuned microsoft/deberta-v3-xsmall checkpoint and predicts
one of:
  - direct
  - intermediate
  - fully_evasive

Usage:
  # Inference on an earnings call Q&A pair:
  python scripts/predict_transformer.py --question "..." --answer "..."

  # Inference with JSON output:
  python scripts/predict_transformer.py --question "..." --answer "..." --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to sys.path
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from finveritas.nlp_evasion.transformer_classifier import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DISCLAIMER_TEXT,
    ID_TO_LABEL,
    LABEL_TO_ID,
    SUPPORTED_CLASSES,
    predict_evasion_transformer,
)

logger = logging.getLogger("predict_transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinVeritas Transformer Question-Evasion Prediction CLI"
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
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Base model architecture (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/transformer_nlp/best_model.pt"),
        help="Path to the saved PyTorch model checkpoint (.pt).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("artifacts/transformer_nlp/tokenizer"),
        help="Directory where tokenizer is stored.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length (default: 384).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda/cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as a structured JSON object.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_name: str,
    checkpoint_path: Path,
    tokenizer_dir: Path,
    device: torch.device,
):
    has_saved_tokenizer = tokenizer_dir.is_dir() and any(
        (tokenizer_dir / f).exists()
        for f in ("vocab.json", "spm.model", "tokenizer_config.json")
    )
    if has_saved_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(SUPPORTED_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    if checkpoint_path.is_file():
        logger.info(f"Loading checkpoint weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            f"Checkpoint {checkpoint_path} not found. Running with base un-fine-tuned weights."
        )

    model.float()
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not (args.question and args.answer):
        logger.error(
            "Both --question and --answer are required for inference.\n"
            "Example: python scripts/predict_transformer.py "
            '--question "What was Q3 revenue?" --answer "Revenue was $5B."'
        )
        return 1

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            tokenizer_dir=args.tokenizer_dir,
            device=device,
        )
    except Exception as err:
        logger.error(f"Failed to load transformer model: {err}")
        return 1

    result = predict_evasion_transformer(
        model=model,
        tokenizer=tokenizer,
        question=args.question,
        answer=args.answer,
        device=device,
        max_length=args.max_length,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    probs = result["uncalibrated_probabilities"]
    print("\n" + "=" * 65)
    print("FINVERITAS QUESTION-EVASION PREDICTION (TRANSFORMER)")
    print("=" * 65)
    print(f"Model:      {result['model_architecture']} ({result['device']})")
    print(f"Question:   {result['question']}")
    print(f"Answer:     {result['answer']}")
    print(f"Prediction: {result['predicted_label'].upper()}")
    print("\nUncalibrated Class Probabilities (Pseudo-Label Agreement):")
    for cls_name, prob in probs.items():
        print(f"  - {cls_name:14s}: {prob*100:5.2f}%")
    if result["is_truncated"]:
        n_tok = result["total_tokens_before_truncation"]
        max_l = result["max_length"]
        print(
            f"\n[WARNING] Input sequence length ({n_tok} tokens) "
            f"exceeded max_length ({max_l}). Answer was truncated."
        )
    print("\n[DISCLAIMER] " + DISCLAIMER_TEXT)
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
