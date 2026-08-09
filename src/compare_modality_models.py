from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wesad_finetuned_common import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_THRESHOLD_OPTIMIZATION_PATH,
    load_selected_threshold,
)
from wesad_modality_common import (
    MODALITY_METRICS_DIR,
    MODALITY_MODEL_ROOT,
    compute_binary_metrics,
    load_model_artifact,
    load_split,
    predict_with_artifact,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BVP/EDA/TEMP/ACC/fusion test performance.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--threshold-json", type=Path, default=DEFAULT_THRESHOLD_OPTIMIZATION_PATH)
    parser.add_argument("--output-json", type=Path, default=MODALITY_METRICS_DIR / "comparison.json")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def evaluate_modality(modality: str, artifact_path: Path, split: str = "test") -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    X, y, _ = load_split(modality, split)
    artifact = load_model_artifact(artifact_path)
    probabilities = predict_with_artifact(artifact, X)
    metrics = compute_binary_metrics(y, probabilities, threshold=float(artifact["threshold"]))
    return probabilities.astype(np.float32), y.astype(np.int64), metrics


def evaluate_fusion(artifact_path: Path, split: str = "test") -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    artifact = load_model_artifact(artifact_path)
    bvp_probabilities_path = MODALITY_MODEL_ROOT / "bvp_probabilities" / f"{split}_probabilities.npy"
    bvp_labels_path = MODALITY_MODEL_ROOT / "bvp_probabilities" / f"{split}_labels.npy"
    if not bvp_probabilities_path.exists() or not bvp_labels_path.exists():
        raise FileNotFoundError("BVP probability cache not found. Run src/compute_bvp_probabilities.py first.")

    bvp_probs = np.load(bvp_probabilities_path)
    y = np.load(bvp_labels_path)
    eda_probs, _, _ = evaluate_modality("eda", MODALITY_MODEL_ROOT / "eda_model.joblib", split)
    temp_probs, _, _ = evaluate_modality("temp", MODALITY_MODEL_ROOT / "temp_model.joblib", split)
    acc_probs, _, _ = evaluate_modality("acc", MODALITY_MODEL_ROOT / "acc_model.joblib", split)
    features = np.column_stack([bvp_probs, eda_probs, temp_probs, acc_probs])
    probabilities = artifact["model"].predict_proba(features)[:, 1]
    metrics = compute_binary_metrics(y, probabilities, threshold=float(artifact["threshold"]))
    return probabilities.astype(np.float32), y.astype(np.int64), metrics


def main() -> None:
    args = parse_args()
    threshold, _ = load_selected_threshold(args.threshold_json)

    print("=" * 70)
    print("VITALAGENT - MODALITY COMPARISON")
    print("=" * 70)

    results = {}
    bvp_probabilities_path = MODALITY_MODEL_ROOT / "bvp_probabilities" / "test_probabilities.npy"
    bvp_labels_path = MODALITY_MODEL_ROOT / "bvp_probabilities" / "test_labels.npy"
    if not bvp_probabilities_path.exists() or not bvp_labels_path.exists():
        raise FileNotFoundError("BVP probability cache not found. Run src/compute_bvp_probabilities.py first.")
    bvp_probs = np.load(bvp_probabilities_path)
    bvp_labels = np.load(bvp_labels_path)
    bvp_metrics = compute_binary_metrics(bvp_labels, bvp_probs, threshold=float(threshold))
    results["bvp"] = {"metrics": bvp_metrics, "threshold": float(threshold)}

    eda_probs, eda_labels, eda_metrics = evaluate_modality("eda", MODALITY_MODEL_ROOT / "eda_model.joblib")
    results["eda"] = {"metrics": eda_metrics, "threshold": float(load_model_artifact(MODALITY_MODEL_ROOT / "eda_model.joblib")["threshold"])}

    temp_probs, temp_labels, temp_metrics = evaluate_modality("temp", MODALITY_MODEL_ROOT / "temp_model.joblib")
    results["temp"] = {"metrics": temp_metrics, "threshold": float(load_model_artifact(MODALITY_MODEL_ROOT / "temp_model.joblib")["threshold"])}

    acc_probs, acc_labels, acc_metrics = evaluate_modality("acc", MODALITY_MODEL_ROOT / "acc_model.joblib")
    results["acc"] = {"metrics": acc_metrics, "threshold": float(load_model_artifact(MODALITY_MODEL_ROOT / "acc_model.joblib")["threshold"])}

    fusion_probs, fusion_labels, fusion_metrics = evaluate_fusion(MODALITY_MODEL_ROOT / "fusion_model.joblib")
    results["fusion"] = {"metrics": fusion_metrics, "threshold": float(load_model_artifact(MODALITY_MODEL_ROOT / "fusion_model.joblib")["threshold"])}

    print("\nModality | Accuracy | Precision | Recall | F1 | ROC-AUC")
    print("-" * 70)
    for modality in ["bvp", "eda", "temp", "acc", "fusion"]:
        metrics = results[modality]["metrics"]
        print(
            f"{modality:>8} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f}"
        )

    save_json(args.output_json, results)
    print(f"\nSaved comparison JSON: {args.output_json}")


if __name__ == "__main__":
    main()
