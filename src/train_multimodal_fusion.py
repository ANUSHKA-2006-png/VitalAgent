from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wesad_finetuned_common import (
    DEFAULT_CHECKPOINT_PATH,
    WesadBvpDataset,
    load_finetuned_checkpoint,
    load_wesad_split,
    predict_probabilities,
)
from wesad_modality_common import (
    MODALITY_METRICS_DIR,
    MODALITY_MODEL_ROOT,
    DEFAULT_SEED,
    class_distribution,
    compute_binary_metrics,
    load_model_artifact,
    load_split,
    predict_with_artifact,
    print_metrics,
    save_json,
    save_model_artifact,
    select_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a late-fusion classifier for BVP+EDA+TEMP+ACC.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact-path", type=Path, default=MODALITY_MODEL_ROOT / "fusion_model.joblib")
    parser.add_argument("--metrics-path", type=Path, default=MODALITY_METRICS_DIR / "fusion_metrics.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def compute_bvp_probabilities_for_splits(
    checkpoint_path: Path,
    device: torch.device,
    allow_download: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    model, _ = load_finetuned_checkpoint(
        checkpoint_path,
        device=device,
        local_files_only=not allow_download,
    )
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        X, y, _ = load_wesad_split(split)
        loader = torch.utils.data.DataLoader(
            WesadBvpDataset(X, y),
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )
        probabilities, labels = predict_probabilities(model, loader, device=device)
        results[split] = (probabilities.astype(np.float32), labels.astype(np.int64))
    return results


def main() -> None:
    args = parse_args()
    artifact_path = args.artifact_path.resolve()
    metrics_path = args.metrics_path.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("VITALAGENT - MULTIMODAL FUSION TRAINING")
    print("=" * 70)

    bvp_probabilities = compute_bvp_probabilities_for_splits(args.checkpoint, device, args.allow_download)
    bvp_train_probs, y_train = bvp_probabilities["train"]
    bvp_val_probs, y_val = bvp_probabilities["val"]
    bvp_test_probs, y_test = bvp_probabilities["test"]

    eda_artifact = load_model_artifact(MODALITY_MODEL_ROOT / "eda_model.joblib")
    temp_artifact = load_model_artifact(MODALITY_MODEL_ROOT / "temp_model.joblib")
    acc_artifact = load_model_artifact(MODALITY_MODEL_ROOT / "acc_model.joblib")

    X_train_eda, y_train_eda, _ = load_split("eda", "train")
    X_val_eda, y_val_eda, _ = load_split("eda", "val")
    X_test_eda, y_test_eda, _ = load_split("eda", "test")
    if not np.array_equal(y_train_eda, y_train):
        raise ValueError("Training labels for BVP and EDA do not align")

    eda_train_probs = predict_with_artifact(eda_artifact, X_train_eda)
    eda_val_probs = predict_with_artifact(eda_artifact, X_val_eda)
    eda_test_probs = predict_with_artifact(eda_artifact, X_test_eda)

    X_train_temp, y_train_temp, _ = load_split("temp", "train")
    X_val_temp, y_val_temp, _ = load_split("temp", "val")
    X_test_temp, y_test_temp, _ = load_split("temp", "test")
    if not np.array_equal(y_train_temp, y_train):
        raise ValueError("Training labels for BVP and TEMP do not align")

    temp_train_probs = predict_with_artifact(temp_artifact, X_train_temp)
    temp_val_probs = predict_with_artifact(temp_artifact, X_val_temp)
    temp_test_probs = predict_with_artifact(temp_artifact, X_test_temp)

    X_train_acc, y_train_acc, _ = load_split("acc", "train")
    X_val_acc, y_val_acc, _ = load_split("acc", "val")
    X_test_acc, y_test_acc, _ = load_split("acc", "test")
    if not np.array_equal(y_train_acc, y_train):
        raise ValueError("Training labels for BVP and ACC do not align")

    acc_train_probs = predict_with_artifact(acc_artifact, X_train_acc)
    acc_val_probs = predict_with_artifact(acc_artifact, X_val_acc)
    acc_test_probs = predict_with_artifact(acc_artifact, X_test_acc)

    X_train_fusion = np.column_stack([bvp_train_probs, eda_train_probs, temp_train_probs, acc_train_probs])
    X_val_fusion = np.column_stack([bvp_val_probs, eda_val_probs, temp_val_probs, acc_val_probs])
    X_test_fusion = np.column_stack([bvp_test_probs, eda_test_probs, temp_test_probs, acc_test_probs])

    X_train_fusion = np.nan_to_num(X_train_fusion, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_fusion = np.nan_to_num(X_val_fusion, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_fusion = np.nan_to_num(X_test_fusion, nan=0.0, posinf=0.0, neginf=0.0)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(X_train_fusion, y_train)
    val_probabilities = model.predict_proba(X_val_fusion)[:, 1]
    threshold, val_metrics = select_threshold(y_val, val_probabilities)
    test_probabilities = model.predict_proba(X_test_fusion)[:, 1]
    test_metrics = compute_binary_metrics(y_test, test_probabilities, threshold=threshold)

    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    artifact = {
        "modality": "fusion",
        "model": model,
        "threshold": float(threshold),
        "validation_metrics": val_metrics,
        "feature_names": ["bvp_probability", "eda_probability", "temp_probability", "acc_probability"],
        "train_class_distribution": class_distribution(y_train),
        "val_class_distribution": class_distribution(y_val),
        "test_class_distribution": class_distribution(y_test),
    }
    save_model_artifact(artifact_path, artifact)
    save_json(
        metrics_path,
        {
            "modality": "fusion",
            "artifact_path": str(artifact_path),
            "threshold": float(threshold),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "class_distribution": {
                "train": class_distribution(y_train),
                "val": class_distribution(y_val),
                "test": class_distribution(y_test),
            },
        },
    )

    print(f"\nSaved model artifact: {artifact_path}")
    print(f"Saved metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
