from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from wesad_finetuned_common import (
    DEFAULT_CHECKPOINT_PATH,
    FINETUNED_DIR,
    PROJECT_ROOT,
    STRESS_EMBEDDING_DIR,
    WesadBvpDataset,
    compute_binary_metrics,
    load_finetuned_checkpoint,
    load_wesad_split,
    predict_probabilities,
    print_metrics,
    save_json,
)


BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "wesad_stress_classifier.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare WESAD baseline RF against fine-tuned MOMENT on the same test split."
    )
    parser.add_argument("--baseline-model", type=Path, default=BASELINE_MODEL_PATH)
    parser.add_argument("--finetuned-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=FINETUNED_DIR / "baseline_vs_finetuned_test.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("VITALAGENT - STRESS MODEL COMPARISON")
    print("=" * 70)
    print("Test subjects: S15, S16, S17")

    X_test, y_test, _ = load_wesad_split("test")
    baseline_embeddings = np.load(STRESS_EMBEDDING_DIR / "X_test_embeddings.npy")
    baseline_y = np.load(STRESS_EMBEDDING_DIR / "y_test.npy")
    if not np.array_equal(y_test, baseline_y):
        raise ValueError("Raw test labels and embedding test labels do not match.")

    print("\nEvaluating baseline: MOMENT embeddings + Random Forest")
    baseline = joblib.load(args.baseline_model)
    baseline_predictions = baseline.predict(baseline_embeddings)
    baseline_probabilities = baseline.predict_proba(baseline_embeddings)[:, 1]
    baseline_metrics = compute_binary_metrics(
        y_test,
        baseline_probabilities,
        threshold=0.5,
    )
    baseline_metrics["confusion_matrix"] = (
        np.asarray(
            [
                [
                    int(((y_test == 0) & (baseline_predictions == 0)).sum()),
                    int(((y_test == 0) & (baseline_predictions == 1)).sum()),
                ],
                [
                    int(((y_test == 1) & (baseline_predictions == 0)).sum()),
                    int(((y_test == 1) & (baseline_predictions == 1)).sum()),
                ],
            ]
        )
        .astype(int)
        .tolist()
    )
    print_metrics("Baseline Test", baseline_metrics)

    print("\nEvaluating fine-tuned: MOMENT + stress classification head")
    model, checkpoint = load_finetuned_checkpoint(
        args.finetuned_checkpoint,
        device=device,
        local_files_only=not args.allow_download,
    )
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    loader = DataLoader(
        WesadBvpDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    fine_probabilities, labels = predict_probabilities(model, loader, device=device)
    fine_metrics = compute_binary_metrics(
        labels,
        fine_probabilities,
        threshold=threshold,
    )
    print_metrics("Fine-tuned Test", fine_metrics)

    comparison = {
        "f1_improved": fine_metrics["f1"] > baseline_metrics["f1"],
        "roc_auc_improved": (
            fine_metrics["roc_auc"] is not None
            and baseline_metrics["roc_auc"] is not None
            and fine_metrics["roc_auc"] > baseline_metrics["roc_auc"]
        ),
        "recall_improved": fine_metrics["recall"] > baseline_metrics["recall"],
        "accuracy_changed": fine_metrics["accuracy"] - baseline_metrics["accuracy"],
        "recommended_model": (
            "fine_tuned_moment"
            if fine_metrics["f1"] > baseline_metrics["f1"]
            else "baseline_random_forest"
        ),
    }

    print("\nComparison")
    print("----------")
    print(f"Fine-tuning improved F1: {comparison['f1_improved']}")
    print(f"Fine-tuning improved ROC-AUC: {comparison['roc_auc_improved']}")
    print(f"Recall improved: {comparison['recall_improved']}")
    print(f"Accuracy change: {comparison['accuracy_changed']:+.4f}")
    print(f"Recommended final stress model: {comparison['recommended_model']}")

    payload = {
        "test_subjects": ["S15", "S16", "S17"],
        "baseline": {
            "name": "MOMENT embeddings + Random Forest",
            "model_path": args.baseline_model,
            "metrics": baseline_metrics,
        },
        "fine_tuned": {
            "name": "MOMENT + stress classification head",
            "checkpoint": args.finetuned_checkpoint,
            "threshold": threshold,
            "metrics": fine_metrics,
        },
        "comparison": comparison,
    }
    save_json(args.save_json, payload)
    print(f"\nSaved comparison: {args.save_json}")


if __name__ == "__main__":
    main()
