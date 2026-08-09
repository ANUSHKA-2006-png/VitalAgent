from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from momentfm import MOMENTPipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRESS_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "stress" / "splits"
STRESS_EMBEDDING_DIR = (
    PROJECT_ROOT / "data" / "processed" / "stress" / "embeddings"
)
FINETUNED_DIR = PROJECT_ROOT / "models" / "wesad" / "finetuned"
DEFAULT_CHECKPOINT_PATH = FINETUNED_DIR / "moment_stress_classifier.pt"
DEFAULT_METRICS_PATH = FINETUNED_DIR / "metrics.json"
DEFAULT_THRESHOLD_OPTIMIZATION_PATH = FINETUNED_DIR / "threshold_optimization.json"
MOMENT_MODEL_ID = "AutonLab/MOMENT-1-large"
CLASS_NAMES = ("NON-STRESS", "STRESS")
EXPECTED_SUBJECTS = {
    "train": {"S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"},
    "val": {"S13", "S14"},
    "test": {"S15", "S16", "S17"},
}


class WesadBvpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = None if y is None else np.asarray(y, dtype=np.int64)

        if self.X.ndim != 2 or self.X.shape[1] != 512:
            raise ValueError(f"Expected X shape (n, 512), got {self.X.shape}.")
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of rows.")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        signal = torch.from_numpy(self.X[index]).unsqueeze(0)
        input_mask = torch.ones(512, dtype=torch.float32)

        if self.y is None:
            return signal, input_mask

        label = torch.tensor(self.y[index], dtype=torch.long)
        return signal, input_mask, label


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_wesad_split(split: str):
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    X = np.load(STRESS_SPLIT_DIR / f"X_{split}.npy")
    y = np.load(STRESS_SPLIT_DIR / f"y_{split}.npy")
    metadata = pd.read_csv(STRESS_SPLIT_DIR / f"metadata_{split}.csv")
    return X, y, metadata


def verify_wesad_split(split: str, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    subjects = set(metadata["subject"].astype(str))
    labels = set(np.unique(y).astype(int))
    metadata_labels = set(metadata["stress_label"].unique().astype(int))

    checks = {
        "split": split,
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "y_shape": list(y.shape),
        "y_dtype": str(y.dtype),
        "subjects": sorted(subjects),
        "class_distribution": class_distribution(y),
        "is_finite": bool(np.isfinite(X).all()),
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
    }

    if X.ndim != 2 or X.shape[1] != 512:
        raise ValueError(f"{split}: expected X shape (n, 512), got {X.shape}.")
    if len(X) != len(y) or len(y) != len(metadata):
        raise ValueError(f"{split}: X, y, and metadata row counts do not match.")
    if labels - {0, 1}:
        raise ValueError(f"{split}: labels must be 0/1, got {sorted(labels)}.")
    if metadata_labels - {0, 1}:
        raise ValueError(
            f"{split}: metadata stress_label must be 0/1, got {sorted(metadata_labels)}."
        )
    if not np.array_equal(y.astype(int), metadata["stress_label"].to_numpy(dtype=int)):
        raise ValueError(f"{split}: y does not match metadata stress_label.")
    if not checks["is_finite"]:
        raise ValueError(f"{split}: X contains NaN or Inf values.")
    if split in EXPECTED_SUBJECTS and subjects != EXPECTED_SUBJECTS[split]:
        raise ValueError(
            f"{split}: subjects {sorted(subjects)} do not match expected "
            f"{sorted(EXPECTED_SUBJECTS[split])}."
        )

    return checks


def verify_no_subject_overlap(metadata_by_split: dict[str, pd.DataFrame]):
    subjects = {
        split: set(metadata["subject"].astype(str))
        for split, metadata in metadata_by_split.items()
    }
    overlaps = {
        "train_val": sorted(subjects.get("train", set()) & subjects.get("val", set())),
        "train_test": sorted(
            subjects.get("train", set()) & subjects.get("test", set())
        ),
        "val_test": sorted(subjects.get("val", set()) & subjects.get("test", set())),
    }
    if any(overlaps.values()):
        raise ValueError(f"Subject overlap detected: {overlaps}")
    return overlaps


def class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def stratified_subset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame | None,
    max_samples: int | None,
    seed: int,
):
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return X, y, metadata

    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)

    selected: list[int] = []
    remaining = int(max_samples)
    for class_value, count in zip(classes, counts):
        take = max(1, int(round(max_samples * (count / len(y)))))
        take = min(take, int(count), remaining)
        class_indices = np.where(y == class_value)[0]
        selected.extend(rng.choice(class_indices, size=take, replace=False).tolist())
        remaining -= take

    if remaining > 0:
        unused = np.setdiff1d(np.arange(len(y)), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(unused, size=min(remaining, len(unused)), replace=False))

    selected = np.asarray(sorted(selected[:max_samples]), dtype=np.int64)
    metadata_subset = None if metadata is None else metadata.iloc[selected].reset_index(drop=True)
    return X[selected], y[selected], metadata_subset


def build_moment_model_kwargs(stage: str, head_dropout: float) -> dict[str, Any]:
    if stage not in {"head", "encoder", "full"}:
        raise ValueError("stage must be one of: head, encoder, full")

    model_kwargs: dict[str, Any] = {
        "task_name": "classification",
        "n_channels": 1,
        "num_class": 2,
        "head_dropout": float(head_dropout),
        "reduction": "concat",
    }

    if stage == "head":
        model_kwargs.update({"freeze_embedder": True, "freeze_encoder": True})
    elif stage == "encoder":
        model_kwargs.update({"freeze_embedder": True, "freeze_encoder": False})
    else:
        model_kwargs.update({"freeze_embedder": False, "freeze_encoder": False})

    return model_kwargs


def load_fresh_moment_classifier(
    *,
    stage: str = "head",
    head_dropout: float = 0.1,
    local_files_only: bool = True,
    device: str | torch.device = "cpu",
):
    model_kwargs = build_moment_model_kwargs(stage=stage, head_dropout=head_dropout)
    model = MOMENTPipeline.from_pretrained(
        MOMENT_MODEL_ID,
        local_files_only=local_files_only,
        model_kwargs=dict(model_kwargs),
    )
    model.init()
    model.to(device)
    return model, model_kwargs


def load_finetuned_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    *,
    device: str | torch.device = "cpu",
    local_files_only: bool = True,
):
    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_kwargs = checkpoint["model_kwargs"]

    model = MOMENTPipeline.from_pretrained(
        checkpoint.get("moment_model_id", MOMENT_MODEL_ID),
        local_files_only=local_files_only,
        model_kwargs=dict(model_kwargs),
    )
    model.init()

    checkpoint_type = checkpoint.get("checkpoint_type", "head_only")
    if checkpoint_type == "head_only":
        model.head.load_state_dict(checkpoint["head_state_dict"])
    elif checkpoint_type == "full_state_dict":
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Unknown checkpoint_type: {checkpoint_type}")

    model.to(device)
    model.eval()
    return model, checkpoint


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Both classes must be present. Got counts {counts.tolist()}.")
    weights = counts.sum() / (2.0 * counts)
    return weights.astype(np.float32)


def predict_probabilities(
    model: torch.nn.Module,
    loader,
    *,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                signals, input_mask, batch_labels = batch
                labels.append(batch_labels.numpy())
            else:
                signals, input_mask = batch

            signals = signals.to(device)
            input_mask = input_mask.to(device)
            output = model(x_enc=signals, input_mask=input_mask)
            batch_probabilities = torch.softmax(output.logits, dim=1)[:, 1]
            probabilities.append(batch_probabilities.cpu().numpy())

    y_true = None if not labels else np.concatenate(labels, axis=0)
    return np.concatenate(probabilities, axis=0), y_true


def predict_one_window(
    model: torch.nn.Module,
    window: np.ndarray,
    *,
    device: str | torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    window = np.asarray(window, dtype=np.float32)
    if window.shape != (512,):
        raise ValueError(f"Expected a 512-sample BVP window, got {window.shape}.")
    if not np.isfinite(window).all():
        raise ValueError("Input BVP window contains NaN or Inf values.")

    signal = torch.from_numpy(window).unsqueeze(0).unsqueeze(0).to(device)
    input_mask = torch.ones((1, 512), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        output = model(x_enc=signal, input_mask=input_mask)
        probabilities = torch.softmax(output.logits, dim=1)[0].cpu().numpy()

    stress_probability = float(probabilities[1])
    prediction = int(stress_probability >= threshold)
    return {
        "prediction": prediction,
        "prediction_text": CLASS_NAMES[prediction],
        "non_stress_probability": float(probabilities[0]),
        "stress_probability": stress_probability,
        "threshold": float(threshold),
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    stress_probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    stress_probabilities = np.asarray(stress_probabilities, dtype=np.float32)
    y_pred = (stress_probabilities >= threshold).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
        "threshold": float(threshold),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, stress_probabilities))

    return metrics


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    roc_auc = metrics.get("roc_auc")
    print(f"ROC-AUC  : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC  : n/a")
    print("Confusion Matrix:")
    print(np.asarray(metrics["confusion_matrix"], dtype=int))


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def load_json(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_selected_threshold(
    path: str | Path = DEFAULT_THRESHOLD_OPTIMIZATION_PATH,
) -> tuple[float, dict[str, Any]]:
    payload = load_json(path)
    if "selected_threshold" not in payload:
        raise ValueError(f"{path} does not contain selected_threshold.")
    return float(payload["selected_threshold"]), payload


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
        handle.write("\n")
