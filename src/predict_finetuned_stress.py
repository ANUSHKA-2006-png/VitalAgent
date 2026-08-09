from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wesad_finetuned_common import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    load_finetuned_checkpoint,
    load_wesad_split,
    predict_one_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict stress for one 512-sample BVP window with fine-tuned MOMENT."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--input-npy", type=Path, default=None)
    parser.add_argument("--true-label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_input(args: argparse.Namespace):
    if args.input_npy is not None:
        window = np.load(args.input_npy)
        if window.ndim == 2 and window.shape[0] == 1:
            window = window[0]
        return np.asarray(window, dtype=np.float32), args.true_label, str(args.input_npy)

    X, y, _ = load_wesad_split(args.split)
    if args.sample_index < 0 or args.sample_index >= len(X):
        raise IndexError(
            f"sample-index must be between 0 and {len(X) - 1} for {args.split}."
        )
    return X[args.sample_index], int(y[args.sample_index]), (
        f"{args.split} sample {args.sample_index}"
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("VITALAGENT - FINE-TUNED STRESS PREDICTION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model, checkpoint = load_finetuned_checkpoint(
        args.checkpoint,
        device=device,
        local_files_only=not args.allow_download,
    )
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )

    window, true_label, source = load_input(args)
    result = predict_one_window(model, window, device=device, threshold=threshold)

    print(f"\nInput: {source}")
    print(f"Input shape: {window.shape}")
    if true_label is not None:
        print(f"True label: {CLASS_NAMES[int(true_label)]}")

    print("\nPrediction:", result["prediction_text"])
    print(f"Stress probability: {result['stress_probability'] * 100:.2f}%")
    print(f"Non-stress probability: {result['non_stress_probability'] * 100:.2f}%")

    if true_label is not None:
        is_correct = int(true_label) == int(result["prediction"])
        print("Correct:", "yes" if is_correct else "no")


if __name__ == "__main__":
    main()
