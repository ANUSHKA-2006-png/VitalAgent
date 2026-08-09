from __future__ import annotations

import argparse
from pathlib import Path

from wesad_modality_common import (
    MODALITY_METRICS_DIR,
    MODALITY_MODEL_ROOT,
    DEFAULT_SEED,
    build_feature_matrix,
    class_distribution,
    compute_binary_metrics,
    load_split,
    print_metrics,
    save_json,
    save_model_artifact,
    train_candidate_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TEMP-only WESAD stress classifier.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact-path", type=Path, default=MODALITY_MODEL_ROOT / "temp_model.joblib")
    parser.add_argument("--metrics-path", type=Path, default=MODALITY_METRICS_DIR / "temp_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = args.artifact_path.resolve()
    metrics_path = args.metrics_path.resolve()

    print("=" * 70)
    print("VITALAGENT - TEMP STRESS MODEL TRAINING")
    print("=" * 70)

    X_train, y_train, _ = load_split("temp", "train")
    X_val, y_val, _ = load_split("temp", "val")
    X_test, y_test, _ = load_split("temp", "test")

    print(f"Train: {X_train.shape} / labels {class_distribution(y_train)}")
    print(f"Val:   {X_val.shape} / labels {class_distribution(y_val)}")
    print(f"Test:  {X_test.shape} / labels {class_distribution(y_test)}")

    X_train_features = build_feature_matrix(X_train, "temp")
    X_val_features = build_feature_matrix(X_val, "temp")
    X_test_features = build_feature_matrix(X_test, "temp")

    model_name, model, threshold, val_metrics = train_candidate_models(
        X_train_features,
        y_train,
        X_val_features,
        y_val,
        modality="temp",
        seed=args.seed,
    )

    test_probabilities = model.predict_proba(X_test_features)[:, 1]
    test_metrics = compute_binary_metrics(y_test, test_probabilities, threshold=threshold)
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    artifact = {
        "modality": "temp",
        "model": model,
        "selected_model": model_name,
        "threshold": float(threshold),
        "validation_metrics": val_metrics,
        "train_class_distribution": class_distribution(y_train),
        "val_class_distribution": class_distribution(y_val),
        "test_class_distribution": class_distribution(y_test),
    }
    save_model_artifact(artifact_path, artifact)
    save_json(
        metrics_path,
        {
            "modality": "temp",
            "artifact_path": str(artifact_path),
            "selected_model": model_name,
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
