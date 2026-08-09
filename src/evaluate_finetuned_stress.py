from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wesad_finetuned_common import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_THRESHOLD_OPTIMIZATION_PATH,
    FINETUNED_DIR,
    WesadBvpDataset,
    compute_binary_metrics,
    load_finetuned_checkpoint,
    load_selected_threshold,
    load_wesad_split,
    predict_probabilities,
    print_metrics,
    resolve_path,
    save_json,
    verify_wesad_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned MOMENT WESAD stress classifier."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-json",
        type=Path,
        default=DEFAULT_THRESHOLD_OPTIMIZATION_PATH,
        help=(
            "Validation threshold JSON produced by optimize_finetuned_threshold.py. "
            "Used when --threshold is not supplied."
        ),
    )
    parser.add_argument(
        "--ignore-threshold-json",
        action="store_true",
        help="Fall back to the checkpoint threshold when --threshold is not supplied.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def resolve_threshold(
    args: argparse.Namespace,
    checkpoint: dict,
) -> tuple[float, str, dict | None]:
    if args.threshold is not None:
        return float(args.threshold), "command-line --threshold", None

    threshold_path = resolve_path(args.threshold_json)
    if not args.ignore_threshold_json and threshold_path.exists():
        threshold, threshold_payload = load_selected_threshold(threshold_path)
        return threshold, f"validation threshold JSON ({threshold_path})", threshold_payload

    return float(checkpoint.get("threshold", 0.5)), "checkpoint threshold", None


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("VITALAGENT - FINE-TUNED WESAD STRESS EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    X, y, metadata = load_wesad_split(args.split)
    checks = verify_wesad_split(args.split, X, y, metadata)
    print(f"\nInput X: {X.shape}")
    print(f"Input y: {y.shape}")
    print(f"Subjects: {checks['subjects']}")
    print(f"Class distribution: {checks['class_distribution']}")

    model, checkpoint = load_finetuned_checkpoint(
        args.checkpoint,
        device=device,
        local_files_only=not args.allow_download,
    )
    threshold, threshold_source, threshold_payload = resolve_threshold(
        args,
        checkpoint,
    )
    print(f"Threshold: {threshold:.4f}")
    print(f"Threshold source: {threshold_source}")

    loader = DataLoader(
        WesadBvpDataset(X, y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities, labels = predict_probabilities(model, loader, device=device)
    metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)

    print_metrics(f"{args.split.upper()} Metrics", metrics)

    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "split_checks": checks,
        "metrics": metrics,
    }
    if threshold_payload is not None:
        payload["threshold_optimization"] = {
            "path": args.threshold_json,
            "selection_split": threshold_payload.get("selection_split"),
            "selection_subjects": threshold_payload.get("selection_subjects"),
            "selection_metric": threshold_payload.get("selection_metric"),
            "selected_validation_metrics": threshold_payload.get(
                "selected_validation_metrics"
            ),
        }

    output_path = args.save_json
    if output_path is None:
        output_path = FINETUNED_DIR / f"{args.split}_metrics.json"
    save_json(output_path, payload)
    print(f"\nSaved metrics: {output_path}")


if __name__ == "__main__":
    main()
