"""
FinVeritas: Transformer-based Earnings-Call Question-Evasion Classifier.

Fine-tunes microsoft/deberta-v3-xsmall for three-class sequence classification:
  - direct
  - intermediate
  - fully_evasive

Input format:
  Sequence 1: Analyst Question
  Sequence 2: Executive Answer

All evaluations use Eva-4B-V2 model-generated pseudo-labels under strict,
leakage-free normalized question group disjoint partitions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import sklearn.metrics
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from finveritas.nlp_evasion.baselines import (
    create_leakage_aware_split,
    normalize_question_for_grouping,
)

logger = logging.getLogger(__name__)

SUPPORTED_CLASSES: tuple[str, ...] = ("direct", "intermediate", "fully_evasive")
LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(SUPPORTED_CLASSES)}
ID_TO_LABEL: dict[int, str] = {i: name for i, name in enumerate(SUPPORTED_CLASSES)}
DEFAULT_MODEL_NAME: str = "microsoft/deberta-v3-xsmall"
DEFAULT_MAX_LENGTH: int = 256
DISCLAIMER_TEXT: str = (
    "These classification scores measure empirical agreement with Eva-4B-V2 "
    "model-generated pseudo-labels on EvasionBench, NOT independently verified "
    "human ground-truth evasion."
)


@dataclass
class TruncationStats:
    total_samples: int = 0
    truncated_samples: int = 0
    truncation_rate: float = 0.0
    max_length_configured: int = DEFAULT_MAX_LENGTH
    notes: str = (
        "Pairs exceeding max_length tokens have executive answer tokens truncated from the end."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransformerModelMetrics:
    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    class_names: list[str] = field(default_factory=lambda: list(SUPPORTED_CLASSES))
    train_wall_time_seconds: float = 0.0
    eval_wall_time_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    truncation_stats: dict[str, Any] = field(default_factory=dict)
    hardware_device: str = "cpu"
    epochs_trained: int = 0
    batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = 2e-5
    scientific_disclaimer: str = DISCLAIMER_TEXT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class EvasionQAPairDataset(Dataset):
    """
    PyTorch Dataset for sequence classification.
    Modes:
      - 'qa': Sequence pair [Question, Answer]
      - 'question_only': Sequence 1 only [Question]
      - 'answer_only': Sequence 1 only [Answer]
    """

    def __init__(
        self,
        questions: list[str],
        answers: list[str],
        labels: list[str] | list[int] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        input_mode: str = "qa",
    ) -> None:
        if input_mode not in ("qa", "question_only", "answer_only"):
            raise ValueError(
                f"Unknown input_mode '{input_mode}'. "
                "Must be 'qa', 'question_only', or 'answer_only'."
            )
        if len(questions) != len(answers):
            raise ValueError(
                f"Mismatch: questions ({len(questions)}) vs answers ({len(answers)})"
            )
        if labels is not None and len(labels) != len(questions):
            raise ValueError(
                f"Mismatch: labels ({len(labels)}) vs questions ({len(questions)})"
            )

        self.questions = questions
        self.answers = answers
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.input_mode = input_mode

        if labels is not None:
            if isinstance(labels[0], str):
                self.label_ids = [LABEL_TO_ID[lbl] for lbl in labels]
            else:
                self.label_ids = [int(lbl) for lbl in labels]
        else:
            self.label_ids = None

        # Precompute truncation statistics
        self.truncation_stats = self._compute_truncation_stats()

    def _compute_truncation_stats(self) -> TruncationStats:
        if self.tokenizer is None or len(self.questions) == 0:
            return TruncationStats(
                total_samples=len(self.questions),
                truncated_samples=0,
                truncation_rate=0.0,
                max_length_configured=self.max_length,
            )

        truncated_count = 0
        for q, a in zip(self.questions, self.answers):
            if self.input_mode == "qa":
                enc = self.tokenizer.encode(q, a, truncation=False)
            elif self.input_mode == "question_only":
                enc = self.tokenizer.encode(q, truncation=False)
            elif self.input_mode == "answer_only":
                enc = self.tokenizer.encode(a, truncation=False)
            else:
                enc = []

            if len(enc) > self.max_length:
                truncated_count += 1

        total = len(self.questions)
        rate = round(truncated_count / total, 4) if total > 0 else 0.0
        return TruncationStats(
            total_samples=total,
            truncated_samples=truncated_count,
            truncation_rate=rate,
            max_length_configured=self.max_length,
        )

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        q = str(self.questions[idx])
        a = str(self.answers[idx])

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required to tokenize items.")

        if self.input_mode == "qa":
            encoding = self.tokenizer(
                q,
                a,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
        elif self.input_mode == "question_only":
            encoding = self.tokenizer(
                q,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
        elif self.input_mode == "answer_only":
            encoding = self.tokenizer(
                a,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = torch.tensor(encoding["token_type_ids"], dtype=torch.long)

        if self.label_ids is not None:
            item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)

        return item


def collate_qa_batch(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int = 0
) -> dict[str, torch.Tensor]:
    """Dynamically pads input_ids and attention_mask to the longest sequence in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    has_token_type = "token_type_ids" in batch[0]
    token_type_ids = (
        torch.zeros((batch_size, max_len), dtype=torch.long) if has_token_type else None
    )

    labels = []
    has_labels = "labels" in batch[0]

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        if has_token_type and token_type_ids is not None:
            token_type_ids[i, :seq_len] = item["token_type_ids"]
        if has_labels:
            labels.append(item["labels"])

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if token_type_ids is not None:
        result["token_type_ids"] = token_type_ids
    if has_labels:
        result["labels"] = torch.stack(labels)

    return result


def load_and_split_evasion_data(
    parquet_path: Path | str,
    random_state: int = 42,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    """
    Loads EvasionBench parquet and creates leakage-free partitions:
      1. Fixed grouped test partition using seed 42 (matching TF-IDF baseline fold 0).
      2. Grouped train/validation split of the remaining data using StratifiedGroupKFold.
      3. Strict assertion of ZERO normalized-question overlap across (train, val, test).
      4. Calculates class weights strictly from the training partition.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"EvasionBench parquet file not found at: {path}")

    table = pq.read_table(
        path,
        columns=["question", "answer", "eva4b_label"],
    )
    df_raw = table.to_pydict()

    questions = [str(q) for q in df_raw["question"]]
    answers = [str(a) for a in df_raw["answer"]]
    labels = [str(lbl) for lbl in df_raw["eva4b_label"]]

    # Validate classes
    for lbl in labels:
        if lbl not in SUPPORTED_CLASSES:
            raise ValueError(f"Unknown label '{lbl}' in dataset. Expected {SUPPORTED_CLASSES}")

    # Step 1: Base test partition (StratifiedGroupKFold fold 0, matching checkpoint 02/03)
    train_val_idx, test_idx, base_meta = create_leakage_aware_split(
        questions=questions,
        labels=labels,
        n_splits=5,
        random_state=random_state,
        fold_to_use=0,
    )

    # Step 2: Split train_val into train and validation with normalized-question grouping
    train_val_questions = [questions[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]
    train_val_groups = [normalize_question_for_grouping(q) for q in train_val_questions]

    n_val_splits = int(round(1.0 / val_fraction))
    sgkf_val = StratifiedGroupKFold(
        n_splits=n_val_splits, shuffle=True, random_state=random_state
    )
    val_splits = list(
        sgkf_val.split(train_val_questions, train_val_labels, groups=train_val_groups)
    )

    inner_train_idx, inner_val_idx = val_splits[0]
    final_train_idx = train_val_idx[inner_train_idx]
    final_val_idx = train_val_idx[inner_val_idx]

    # Step 3: Zero-leakage verification across all 3 sets
    norm_all = [normalize_question_for_grouping(q) for q in questions]
    train_groups = set(norm_all[i] for i in final_train_idx)
    val_groups = set(norm_all[i] for i in final_val_idx)
    test_groups = set(norm_all[i] for i in test_idx)

    overlap_train_val = train_groups & val_groups
    overlap_train_test = train_groups & test_groups
    overlap_val_test = val_groups & test_groups

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Leakage detected! "
            f"Train/Val overlap: {len(overlap_train_val)}, "
            f"Train/Test overlap: {len(overlap_train_test)}, "
            f"Val/Test overlap: {len(overlap_val_test)}"
        )

    # Step 4: Class weights calculated strictly on the training partition
    train_labels = [labels[i] for i in final_train_idx]
    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(SUPPORTED_CLASSES)),
        y=np.array(train_labels),
    )
    class_weights_dict = {
        cls_name: float(w) for cls_name, w in zip(SUPPORTED_CLASSES, computed_weights)
    }

    return {
        "train": {
            "questions": [questions[i] for i in final_train_idx],
            "answers": [answers[i] for i in final_train_idx],
            "labels": [labels[i] for i in final_train_idx],
            "indices": final_train_idx.tolist(),
        },
        "val": {
            "questions": [questions[i] for i in final_val_idx],
            "answers": [answers[i] for i in final_val_idx],
            "labels": [labels[i] for i in final_val_idx],
            "indices": final_val_idx.tolist(),
        },
        "test": {
            "questions": [questions[i] for i in test_idx],
            "answers": [answers[i] for i in test_idx],
            "labels": [labels[i] for i in test_idx],
            "indices": test_idx.tolist(),
        },
        "class_weights": class_weights_dict,
        "metadata": {
            "total_samples": len(questions),
            "train_samples": len(final_train_idx),
            "val_samples": len(final_val_idx),
            "test_samples": len(test_idx),
            "unique_groups_train": len(train_groups),
            "unique_groups_val": len(val_groups),
            "unique_groups_test": len(test_groups),
            "group_overlap_train_val": 0,
            "group_overlap_train_test": 0,
            "group_overlap_val_test": 0,
            "random_state": random_state,
        },
    }


def compute_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    class_names: tuple[str, ...] = SUPPORTED_CLASSES,
) -> dict[str, Any]:
    """Computes accuracy, macro-F1, weighted-F1, per-class metrics, and confusion matrix."""
    acc = float(sklearn.metrics.accuracy_score(y_true, y_pred))
    macro_f1 = float(
        sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        sklearn.metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    p_per, r_per, f1_per, sup_per = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=list(class_names), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i]),
        }

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=list(class_names)).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def run_forward_backward_smoke_test(
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    """
    Executes a tiny forward-pass and backward-pass smoke test on a synthetic pair
    before launching full training to verify environment, architecture, and memory.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running preflight forward/backward smoke test on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(SUPPORTED_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.float()
    model.to(device)
    model.train()

    test_q = ["What was Q3 revenue?", "Why did operating margin decline?"]
    test_a = ["Revenue grew 8% year-over-year to $4.2B.", "We decline to comment on margins."]
    labels = torch.tensor([0, 2], dtype=torch.long, device=device)

    inputs = tokenizer(
        test_q,
        test_a,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    elapsed = time.time() - t0

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    logger.info(
        f"Smoke test passed! Loss={loss.item():.4f}, "
        f"Elapsed={elapsed:.3f}s, PeakVRAM={peak_mem_mb:.1f}MB"
    )
    return {
        "smoke_test_passed": True,
        "device": str(device),
        "loss": round(float(loss.item()), 4),
        "smoke_time_seconds": round(elapsed, 4),
        "peak_vram_mb": round(peak_mem_mb, 2),
    }


def train_transformer_model(
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Path | str = "artifacts/transformer_nlp",
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = DEFAULT_MAX_LENGTH,
    weight_decay: float = 0.01,
    seed: int = 42,
    use_class_weights: bool = True,
    device: torch.device | None = None,
    smoke_test_only: bool = False,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    input_mode: str = "qa",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, TransformerModelMetrics]:
    """
    Fine-tunes the transformer model on the train partition, selecting the best
    checkpoint based on validation Macro-F1 only.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Preflight smoke test
    run_forward_backward_smoke_test(model_name=model_name, device=device, max_length=max_length)
    if smoke_test_only:
        logger.info("smoke_test_only=True requested. Exiting before full training.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        dummy_metrics = TransformerModelMetrics(
            model_name=model_name,
            accuracy=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            per_class={},
            confusion_matrix=[],
            hardware_device=str(device),
        )
        return model, tokenizer, dummy_metrics

    # 2. Tokenizer and Datasets
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_q = train_data["questions"]
    train_a = train_data["answers"]
    train_y = train_data["labels"]
    if max_train_samples and max_train_samples < len(train_q):
        logger.info(f"Subsampling training set to {max_train_samples} samples")
        train_q = train_q[:max_train_samples]
        train_a = train_a[:max_train_samples]
        train_y = train_y[:max_train_samples]

    val_q = val_data["questions"]
    val_a = val_data["answers"]
    val_y = val_data["labels"]
    if max_val_samples and max_val_samples < len(val_q):
        logger.info(f"Subsampling validation set to {max_val_samples} samples")
        val_q = val_q[:max_val_samples]
        val_a = val_a[:max_val_samples]
        val_y = val_y[:max_val_samples]

    train_dataset = EvasionQAPairDataset(
        questions=train_q,
        answers=train_a,
        labels=train_y,
        tokenizer=tokenizer,
        max_length=max_length,
        input_mode=input_mode,
    )
    val_dataset = EvasionQAPairDataset(
        questions=val_q,
        answers=val_a,
        labels=val_y,
        tokenizer=tokenizer,
        max_length=max_length,
        input_mode=input_mode,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_qa_batch(b, pad_token_id=pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_qa_batch(b, pad_token_id=pad_id),
    )

    # 3. Model setup
    logger.info(f"Loading model: {model_name} on {device}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(SUPPORTED_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.float()
    model.to(device)

    # Loss with training class weights if specified
    if use_class_weights:
        computed_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(list(SUPPORTED_CLASSES)),
            y=np.array(train_y),
        )
        weights_tensor = torch.tensor(computed_weights, dtype=torch.float, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
        logger.info(f"Using class-weighted CrossEntropyLoss: {computed_weights}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Precision setup: DeBERTa-v3 uses disentangled relative attention which requires FP32
    # on CUDA for complete numerical stability and to prevent NaN gradients.
    amp_dtype = None
    logger.info(
        f"Starting training: {epochs} epochs, batch_size={batch_size}, "
        f"grad_accum={grad_accum_steps}, device={device}, precision=FP32"
    )

    best_val_macro_f1 = -1.0
    best_checkpoint_path = out_path / "best_model.pt"
    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = (
                batch["token_type_ids"].to(device) if "token_type_ids" in batch else None
            )

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            if amp_dtype is not None:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(**kwargs)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)
                    loss_scaled = loss / grad_accum_steps
            else:
                outputs = model(**kwargs)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                loss_scaled = loss / grad_accum_steps

            loss_scaled.backward()
            total_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                logger.info(
                    f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f} | Avg Loss: {total_loss/(step+1):.4f}"
                )

        # Validation at epoch end
        model.eval()
        val_preds: list[str] = []
        val_trues: list[str] = []

        with torch.no_grad():
            for val_batch in val_loader:
                input_ids = val_batch["input_ids"].to(device)
                attention_mask = val_batch["attention_mask"].to(device)
                labels = val_batch["labels"].to(device)
                token_type_ids = (
                    val_batch["token_type_ids"].to(device)
                    if "token_type_ids" in val_batch
                    else None
                )

                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids

                if amp_dtype is not None:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        outputs = model(**kwargs)
                        preds = torch.argmax(outputs.logits, dim=-1)
                else:
                    outputs = model(**kwargs)
                    preds = torch.argmax(outputs.logits, dim=-1)

                val_preds.extend([ID_TO_LABEL[p.item()] for p in preds])
                val_trues.extend([ID_TO_LABEL[lbl.item()] for lbl in labels])

        val_metrics = compute_metrics(val_trues, val_preds)
        val_macro_f1 = val_metrics["macro_f1"]
        logger.info(
            f"Epoch {epoch} Validation: Macro-F1={val_macro_f1:.4f}, "
            f"Accuracy={val_metrics['accuracy']:.4f}, "
            f"Fully-Evasive F1={val_metrics['per_class']['fully_evasive']['f1']:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            logger.info(
                f"Val Macro-F1 improved to {best_val_macro_f1:.4f}. Saving best checkpoint..."
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": best_val_macro_f1,
                    "model_name": model_name,
                },
                best_checkpoint_path,
            )

    train_wall_time = time.time() - t_train_start

    # Load best weights
    if best_checkpoint_path.is_file():
        logger.info(f"Loading best checkpoint from {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    peak_gpu_mem_mb = 0.0
    if device.type == "cuda":
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Save tokenizer files for reproducible inference
    tokenizer.save_pretrained(out_path / "tokenizer")

    metrics_obj = TransformerModelMetrics(
        model_name=model_name,
        accuracy=val_metrics["accuracy"],
        macro_f1=val_metrics["macro_f1"],
        weighted_f1=val_metrics["weighted_f1"],
        per_class=val_metrics["per_class"],
        confusion_matrix=val_metrics["confusion_matrix"],
        train_wall_time_seconds=round(train_wall_time, 3),
        peak_gpu_memory_mb=round(peak_gpu_mem_mb, 2),
        truncation_stats=train_dataset.truncation_stats.to_dict(),
        hardware_device=str(device),
        epochs_trained=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
    )

    return model, tokenizer, metrics_obj


def evaluate_transformer_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_data: dict[str, Any],
    device: torch.device | None = None,
    batch_size: int = 4,
    max_length: int = DEFAULT_MAX_LENGTH,
    train_wall_time: float = 0.0,
    peak_gpu_memory_mb: float = 0.0,
    input_mode: str = "qa",
) -> TransformerModelMetrics:
    """Evaluates the model on the held-out test partition."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    test_dataset = EvasionQAPairDataset(
        questions=test_data["questions"],
        answers=test_data["answers"],
        labels=test_data["labels"],
        tokenizer=tokenizer,
        max_length=max_length,
        input_mode=input_mode,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_qa_batch(b, pad_token_id=pad_id),
    )

    amp_dtype = None
    y_preds: list[str] = []
    y_trues: list[str] = []

    t_eval_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = (
                batch["token_type_ids"].to(device) if "token_type_ids" in batch else None
            )

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            if amp_dtype is not None:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(**kwargs)
                    preds = torch.argmax(outputs.logits, dim=-1)
            else:
                outputs = model(**kwargs)
                preds = torch.argmax(outputs.logits, dim=-1)

            y_preds.extend([ID_TO_LABEL[p.item()] for p in preds])
            y_trues.extend([ID_TO_LABEL[lbl.item()] for lbl in labels])

    eval_wall_time = time.time() - t_eval_start
    metrics = compute_metrics(y_trues, y_preds)

    if device.type == "cuda" and peak_gpu_memory_mb == 0.0:
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return TransformerModelMetrics(
        model_name=getattr(model.config, "_name_or_path", DEFAULT_MODEL_NAME),
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        per_class=metrics["per_class"],
        confusion_matrix=metrics["confusion_matrix"],
        train_wall_time_seconds=round(train_wall_time, 3),
        eval_wall_time_seconds=round(eval_wall_time, 3),
        peak_gpu_memory_mb=round(peak_gpu_memory_mb, 2),
        truncation_stats=test_dataset.truncation_stats.to_dict(),
        hardware_device=str(device),
    )


def predict_evasion_transformer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    device: torch.device | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    """Single-pair inference with confidence scores and disclaimer."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    encoded_full = tokenizer.encode(question, answer, truncation=False)
    is_truncated = len(encoded_full) > max_length

    inputs = tokenizer(
        question,
        answer,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = ID_TO_LABEL[pred_id]

    prob_dict = {name: round(float(probs[i]), 4) for i, name in enumerate(SUPPORTED_CLASSES)}

    return {
        "question": question,
        "answer": answer,
        "predicted_label": pred_label,
        "uncalibrated_probabilities": prob_dict,
        "is_truncated": is_truncated,
        "total_tokens_before_truncation": len(encoded_full),
        "max_length": max_length,
        "model_architecture": getattr(model.config, "_name_or_path", DEFAULT_MODEL_NAME),
        "device": str(device),
        "disclaimer": DISCLAIMER_TEXT,
    }
