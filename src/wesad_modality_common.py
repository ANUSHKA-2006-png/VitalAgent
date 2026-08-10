from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRESS_ROOT = PROJECT_ROOT / "data" / "processed" / "stress"
MODALITY_MODEL_ROOT = PROJECT_ROOT / "models" / "wesad" / "modality"
MODALITY_METRICS_DIR = MODALITY_MODEL_ROOT / "metrics"
DEFAULT_SEED = 42
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEP = 0.01


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_directory(path: str | Path) -> Path:
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = ensure_directory(path if isinstance(path, Path) else Path(path)) if not Path(path).is_absolute() else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
        handle.write("\n")


def load_json(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_split(modality: str, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    modality = modality.lower()
    split = split.lower()
    if modality not in {"bvp", "eda", "temp", "acc"}:
        raise ValueError("modality must be one of: bvp, eda, temp, acc")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    base_dir = STRESS_ROOT if modality == "bvp" else STRESS_ROOT / modality
    split_dir = base_dir / "splits"
    X = np.load(split_dir / f"X_{split}.npy")
    y = np.load(split_dir / f"y_{split}.npy")
    metadata = pd.read_csv(split_dir / f"metadata_{split}.csv")
    return X, y, metadata


def class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    y_pred = (probabilities >= threshold).astype(np.int64)
    roc_auc: float | None = None
    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
        "threshold": threshold,
    }
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


def threshold_grid(start: float = THRESHOLD_MIN, stop: float = THRESHOLD_MAX, step: float = THRESHOLD_STEP) -> np.ndarray:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    if start > stop:
        raise ValueError("threshold minimum must be less than or equal to maximum")
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    thresholds = start + np.arange(count, dtype=np.float64) * step
    thresholds = thresholds[thresholds <= stop + 1e-9]
    return np.round(thresholds, 10)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_threshold: float | None = None
    best_metrics: dict[str, Any] | None = None
    for threshold in threshold_grid():
        metrics = compute_binary_metrics(y_true, probabilities, threshold=float(threshold))
        if best_metrics is None or best_threshold is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if metrics["f1"] > best_metrics["f1"] + 1e-12:
            best_threshold = float(threshold)
            best_metrics = metrics
        elif abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12 and float(threshold) < best_threshold:
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Failed to select a threshold")
    return best_threshold, best_metrics


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1 or len(y) <= 1:
        return 0.0
    if np.std(x) <= 0 or np.std(y) <= 0:
        return 0.0
    correlation = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return correlation


def _stat_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        raise ValueError("Signal window must not be empty")
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0:
        skew = 0.0
        kurtosis = 0.0
    else:
        centered = values - mean
        skew = float(np.mean(centered**3) / (std**3))
        kurtosis = float(np.mean(centered**4) / (std**4))
    diff = np.diff(values)
    if len(diff) == 0:
        diff_mean = 0.0
        diff_std = 0.0
        diff_abs_mean = 0.0
        diff_abs_max = 0.0
    else:
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff))
        diff_abs_mean = float(np.mean(np.abs(diff)))
        diff_abs_max = float(np.max(np.abs(diff)))
    slopes = np.polyfit(np.arange(len(values)), values, 1)[0]
    features = np.array(
        [
            mean,
            std,
            float(np.min(values)),
            float(np.max(values)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 75)),
            float(np.ptp(values)),
            float(np.sqrt(np.mean(values**2))),
            diff_mean,
            diff_std,
            diff_abs_mean,
            diff_abs_max,
            float(np.mean(np.abs(values))),
            skew,
            kurtosis,
            slopes,
            float(np.sum(values**2)),
            float(np.mean(np.abs(np.diff(values)))) if len(diff) > 0 else 0.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_matrix(X: np.ndarray, modality: str) -> np.ndarray:
    modality = modality.lower()
    if modality in {"eda", "temp"}:
        features = np.stack([_stat_features(window) for window in X], axis=0)
        return features.astype(np.float32)

    if modality == "acc":
        samples = []
        for window in X:
            window = np.asarray(window, dtype=np.float32)
            if window.ndim != 2 or window.shape[1] != 3:
                raise ValueError(f"Expected ACC windows with shape (samples, 3), got {window.shape}")
            axes = [window[:, axis] for axis in range(3)]
            magnitude = np.linalg.norm(window, axis=1)
            axis_features = np.concatenate([_stat_features(axis) for axis in axes])
            magnitude_features = _stat_features(magnitude)
            corr_xy = _safe_correlation(axes[0], axes[1])
            corr_xz = _safe_correlation(axes[0], axes[2])
            corr_yz = _safe_correlation(axes[1], axes[2])
            feature_vector = np.concatenate([axis_features, magnitude_features, np.array([corr_xy, corr_xz, corr_yz], dtype=np.float32)])
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            samples.append(feature_vector)
        return np.stack(samples, axis=0).astype(np.float32)

    if modality == "bvp":
        features = np.stack([_stat_features(window) for window in X], axis=0)
        return features.astype(np.float32)

    raise ValueError(f"Unsupported modality: {modality}")


def train_candidate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    modality: str,
    seed: int = DEFAULT_SEED,
) -> tuple[str, Any, float, dict[str, Any]]:
    candidates = [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=4000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ]

    best_name: str | None = None
    best_model: Any = None
    best_threshold: float | None = None
    best_metrics: dict[str, Any] | None = None

    for name, model in candidates:
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_val)[:, 1]
        threshold, metrics = select_threshold(y_val, probabilities)
        if best_metrics is None or best_threshold is None or metrics["f1"] > best_metrics["f1"] + 1e-12:
            best_name = name
            best_model = model
            best_threshold = threshold
            best_metrics = metrics
        elif abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12 and threshold < best_threshold:
            best_name = name
            best_model = model
            best_threshold = threshold
            best_metrics = metrics

    if best_name is None or best_model is None or best_threshold is None or best_metrics is None:
        raise RuntimeError(f"Failed to train a modality model for {modality}")

    return best_name, best_model, best_threshold, best_metrics


def save_model_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected a dictionary artifact in {path}")
    return artifact


def predict_with_artifact(artifact: dict[str, Any], X: np.ndarray) -> np.ndarray:
    model = artifact["model"]
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    X_features = build_feature_matrix(X, artifact["modality"])
    probabilities = model.predict_proba(X_features)[:, 1]
    return np.asarray(probabilities, dtype=np.float32)


def align_probability(prob: float | np.ndarray, tau: float) -> float | np.ndarray:
    prob_arr = np.asarray(prob, dtype=np.float32)
    tau = tau
    if tau <= 0.0 or tau >= 1.0:
        return prob
    aligned = np.where(
        prob_arr <= tau,
        0.5 * (prob_arr / tau),
        0.5 + 0.5 * ((prob_arr - tau) / (1.0 - tau)),
    )
    aligned = np.clip(aligned, 0.0, 1.0)
    if np.ndim(prob) == 0:
        return float(aligned)
    return aligned


def compute_oof_modality_probabilities(
    modality: str,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    X_train, y_train, metadata_train = load_split(modality, "train")
    subjects = metadata_train["subject"].to_numpy()
    unique_subjects = sorted(np.unique(subjects))

    oof_probabilities = np.zeros(len(y_train), dtype=np.float32)

    for subject in unique_subjects:
        train_idx = np.where(subjects != subject)[0]
        val_idx = np.where(subjects == subject)[0]

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val_subj = X_train[val_idx]
        y_val_subj = y_train[val_idx]

        X_tr_feats = build_feature_matrix(X_tr, modality)
        X_val_feats = build_feature_matrix(X_val_subj, modality)

        _, model, _, _ = train_candidate_models(
            X_tr_feats,
            y_tr,
            X_val_feats,
            y_val_subj,
            modality=modality,
            seed=seed,
        )
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        probs = model.predict_proba(X_val_feats)[:, 1]
        oof_probabilities[val_idx] = probs.astype(np.float32)

    return oof_probabilities
