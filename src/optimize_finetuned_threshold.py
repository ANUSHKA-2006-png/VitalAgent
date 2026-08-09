from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wesad_finetuned_common import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_THRESHOLD_OPTIMIZATION_PATH,
    WesadBvpDataset,
    compute_binary_metrics,
    load_finetuned_checkpoint,
    load_wesad_split,
    predict_probabilities,
    print_metrics,
    save_json,
    verify_wesad_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a MOMENT stress decision threshold using only the WESAD "
            "validation split."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=DEFAULT_THRESHOLD_OPTIMIZATION_PATH,
    )
    return parser.parse_args()


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("--threshold-step must be positive.")
    if start > stop:
        raise ValueError("--threshold-min must be less than or equal to --threshold-max.")

    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    thresholds = start + np.arange(count, dtype=np.float64) * step
    thresholds = thresholds[thresholds <= stop + 1e-9]
    return np.round(thresholds, 10)


def threshold_metrics(
    y_true: np.ndarray,
    stress_probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, Any]]:
    results = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(
            y_true,
            stress_probabilities,
            threshold=float(threshold),
        )
        results.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
    return results


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("VITALAGENT - VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("Selection split: val")
    print(f"Threshold range: {args.threshold_min:.2f} to {args.threshold_max:.2f}")
    print(f"Threshold step: {args.threshold_step:.2f}")

    X_val, y_val, metadata_val = load_wesad_split("val")
    checks = verify_wesad_split("val", X_val, y_val, metadata_val)
    print(f"\nValidation X: {X_val.shape}")
    print(f"Validation y: {y_val.shape}")
    print(f"Subjects: {checks['subjects']}")
    print(f"Class distribution: {checks['class_distribution']}")

    model, checkpoint = load_finetuned_checkpoint(
        args.checkpoint,
        device=device,
        local_files_only=not args.allow_download,
    )
    loader = DataLoader(
        WesadBvpDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    probabilities, labels = predict_probabilities(model, loader, device=device)
    if labels is None:
        raise ValueError("Validation labels are required for threshold optimization.")

    thresholds = threshold_grid(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    results = threshold_metrics(labels, probabilities, thresholds)
    selected = max(results, key=lambda item: (item["f1"], -item["threshold"]))
    selected_metrics = compute_binary_metrics(
        labels,
        probabilities,
        threshold=selected["threshold"],
    )

    print_metrics("Selected Validation Threshold", selected_metrics)

    payload = {
        "checkpoint": args.checkpoint,
        "moment_model_id": checkpoint.get("moment_model_id"),
        "selection_split": "val",
        "selection_subjects": checks["subjects"],
        "selection_metric": "validation_f1",
        "tie_breaker": "lowest threshold among exact F1 ties",
        "selected_threshold": selected["threshold"],
        "selected_validation_metrics": selected_metrics,
        "threshold_search": {
            "min": float(args.threshold_min),
            "max": float(args.threshold_max),
            "step": float(args.threshold_step),
            "count": int(len(thresholds)),
        },
        "threshold_results": results,
        "split_checks": checks,
    }
    save_json(args.save_json, payload)

    print(f"\nSelected threshold: {selected['threshold']:.2f}")
    print(f"Validation F1: {selected_metrics['f1']:.4f}")
    print(f"Saved threshold optimization: {args.save_json}")


if __name__ == "__main__":
    main()
