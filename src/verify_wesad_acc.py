from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_wesad_acc import (
    ACC_AXES,
    ACC_FS,
    ACC_WINDOW,
    EDA_METADATA_PATH,
    NON_STRESS_LABELS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SPLIT_SUBJECTS,
    STRESS_LABEL,
    TEMP_METADATA_PATH,
    WINDOW_SECONDS,
    metadata_matches_bvp,
    metadata_matches_low_rate_reference,
)


BVP_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "stress"
BVP_SPLIT_DIR = BVP_PROCESSED_DIR / "splits"
EDA_SPLIT_DIR = EDA_METADATA_PATH.parent / "splits"
TEMP_SPLIT_DIR = TEMP_METADATA_PATH.parent / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify processed WESAD wrist ACC windows.")
    parser.add_argument("--acc-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_dataset(base_dir: Path, split: str | None = None):
    if split is None:
        X_path = base_dir / "X.npy"
        y_path = base_dir / "y.npy"
        metadata_path = base_dir / "metadata.csv"
    else:
        split_dir = base_dir / "splits"
        X_path = split_dir / f"X_{split}.npy"
        y_path = split_dir / f"y_{split}.npy"
        metadata_path = split_dir / f"metadata_{split}.csv"

    X = np.load(X_path)
    y = np.load(y_path)
    metadata = pd.read_csv(metadata_path)
    return X, y, metadata


def array_nan_count(values: np.ndarray) -> int:
    if np.issubdtype(values.dtype, np.number):
        return int(np.isnan(values).sum())
    return 0


def array_inf_count(values: np.ndarray) -> int:
    if np.issubdtype(values.dtype, np.number):
        return int(np.isinf(values).sum())
    return 0


def class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def subject_distribution(metadata: pd.DataFrame) -> dict[str, int]:
    return {
        str(subject): int(count)
        for subject, count in metadata["subject"].value_counts().sort_index().items()
    }


def axis_stats(X: np.ndarray) -> dict[str, list[float]]:
    axes = ("x", "y", "z")
    return {
        "minimum": [float(value) for value in np.min(X, axis=(0, 1))],
        "maximum": [float(value) for value in np.max(X, axis=(0, 1))],
        "mean": [float(value) for value in np.mean(X, axis=(0, 1))],
        "standard_deviation": [float(value) for value in np.std(X, axis=(0, 1))],
        "axis_order": list(axes),
    }


def dataset_summary(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    return {
        "acc_sampling_rate_hz": ACC_FS,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "metadata_shape": list(metadata.shape),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "X_nan_count": array_nan_count(X),
        "y_nan_count": array_nan_count(y),
        "X_inf_count": array_inf_count(X),
        "y_inf_count": array_inf_count(y),
        "minimum": float(np.min(X)),
        "maximum": float(np.max(X)),
        "mean": float(np.mean(X)),
        "standard_deviation": float(np.std(X)),
        "axis_stats": axis_stats(X),
        "class_distribution": class_distribution(y),
        "subject_distribution": subject_distribution(metadata),
        "unique_accepted_wesad_labels": sorted(
            metadata["wesad_label"].astype(int).unique().tolist()
        ),
    }


def print_summary(name: str, summary: dict[str, Any]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"ACC sampling rate: {summary['acc_sampling_rate_hz']} Hz")
    print(f"X shape: {tuple(summary['X_shape'])}")
    print(f"y shape: {tuple(summary['y_shape'])}")
    print(f"metadata shape: {tuple(summary['metadata_shape'])}")
    print(f"X dtype: {summary['X_dtype']}")
    print(f"y dtype: {summary['y_dtype']}")
    print(f"NaN count: X={summary['X_nan_count']} y={summary['y_nan_count']}")
    print(f"Inf count: X={summary['X_inf_count']} y={summary['y_inf_count']}")
    print(f"Minimum: {summary['minimum']:.6f}")
    print(f"Maximum: {summary['maximum']:.6f}")
    print(f"Mean: {summary['mean']:.6f}")
    print(f"Standard deviation: {summary['standard_deviation']:.6f}")
    print(f"Axis order: {summary['axis_stats']['axis_order']}")
    print(f"Axis minimums: {summary['axis_stats']['minimum']}")
    print(f"Axis maximums: {summary['axis_stats']['maximum']}")
    print(f"Axis means: {summary['axis_stats']['mean']}")
    print(f"Axis standard deviations: {summary['axis_stats']['standard_deviation']}")
    print(f"Class distribution: {summary['class_distribution']}")
    print(f"Subject distribution: {summary['subject_distribution']}")
    print(f"Unique accepted WESAD labels: {summary['unique_accepted_wesad_labels']}")


def validate_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    expected_subjects: set[str] | None = None,
) -> None:
    if X.ndim != 3 or X.shape[1:] != (ACC_WINDOW, ACC_AXES):
        raise ValueError(
            f"{name}: expected X shape (n, {ACC_WINDOW}, {ACC_AXES}), got {X.shape}"
        )
    if y.ndim != 1:
        raise ValueError(f"{name}: expected y shape (n,), got {y.shape}")
    if len(X) != len(y) or len(y) != len(metadata):
        raise ValueError(f"{name}: X, y, and metadata row counts do not match.")
    if X.dtype != np.float32:
        raise ValueError(f"{name}: expected X dtype float32, got {X.dtype}")
    if y.dtype != np.int64:
        raise ValueError(f"{name}: expected y dtype int64, got {y.dtype}")
    if array_nan_count(X) or array_inf_count(X):
        raise ValueError(f"{name}: X contains NaN or Inf values.")
    if set(np.unique(y).astype(int)) - {0, 1}:
        raise ValueError(f"{name}: y must contain only 0/1 labels.")
    if not np.array_equal(y.astype(int), metadata["stress_label"].to_numpy(dtype=int)):
        raise ValueError(f"{name}: y does not match metadata stress_label.")

    subjects = set(metadata["subject"].astype(str))
    if expected_subjects is not None and subjects != expected_subjects:
        raise ValueError(
            f"{name}: subjects {sorted(subjects)} do not match expected "
            f"{sorted(expected_subjects)}."
        )

    wesad_labels = set(metadata["wesad_label"].astype(int))
    allowed_labels = {STRESS_LABEL} | set(NON_STRESS_LABELS)
    if wesad_labels - allowed_labels:
        raise ValueError(
            f"{name}: unexpected original WESAD labels: {sorted(wesad_labels - allowed_labels)}"
        )

    stress_rows = metadata["wesad_label"].astype(int) == STRESS_LABEL
    if not np.array_equal(
        stress_rows.astype(int).to_numpy(),
        metadata["stress_label"].astype(int).to_numpy(),
    ):
        raise ValueError(f"{name}: WESAD stress label mapping is inconsistent.")

    durations = metadata["end_time_seconds"] - metadata["start_time_seconds"]
    if not np.allclose(durations.to_numpy(dtype=float), WINDOW_SECONDS):
        raise ValueError(f"{name}: not all metadata windows are {WINDOW_SECONDS} seconds.")

    sample_lengths = metadata["end_sample"] - metadata["start_sample"]
    if not np.all(sample_lengths.to_numpy(dtype=int) == ACC_WINDOW):
        raise ValueError(f"{name}: not all ACC windows contain {ACC_WINDOW} samples.")

    if not np.all(metadata["acc_sampling_rate_hz"].to_numpy(dtype=float) == ACC_FS):
        raise ValueError(f"{name}: unexpected ACC sampling-rate metadata.")
    if not np.all(metadata["acc_axes"].to_numpy(dtype=int) == ACC_AXES):
        raise ValueError(f"{name}: unexpected ACC axis metadata.")


def verify_reference_alignment(acc_dir: Path) -> None:
    _, _, acc_metadata = load_dataset(acc_dir)
    bvp_metadata = pd.read_csv(BVP_PROCESSED_DIR / "metadata.csv")
    eda_metadata = pd.read_csv(EDA_METADATA_PATH)
    temp_metadata = pd.read_csv(TEMP_METADATA_PATH)

    full_bvp_match = metadata_matches_bvp(acc_metadata, bvp_metadata)
    full_eda_match = metadata_matches_low_rate_reference(acc_metadata, eda_metadata)
    full_temp_match = metadata_matches_low_rate_reference(acc_metadata, temp_metadata)

    print("\nReference alignment")
    print("-------------------")
    print(f"Full ACC metadata aligns with existing BVP metadata: {full_bvp_match}")
    print(f"Full ACC metadata aligns with existing EDA metadata: {full_eda_match}")
    print(f"Full ACC metadata aligns with existing TEMP metadata: {full_temp_match}")

    if not full_bvp_match:
        raise ValueError("Full ACC metadata does not align with existing BVP metadata.")
    if not full_eda_match:
        raise ValueError("Full ACC metadata does not align with existing EDA metadata.")
    if not full_temp_match:
        raise ValueError("Full ACC metadata does not align with existing TEMP metadata.")

    split_subjects: dict[str, set[str]] = {}
    for split, expected_subjects_list in SPLIT_SUBJECTS.items():
        _, _, acc_split_metadata = load_dataset(acc_dir, split)
        bvp_split_metadata = pd.read_csv(BVP_SPLIT_DIR / f"metadata_{split}.csv")
        eda_split_metadata = pd.read_csv(EDA_SPLIT_DIR / f"metadata_{split}.csv")
        temp_split_metadata = pd.read_csv(TEMP_SPLIT_DIR / f"metadata_{split}.csv")

        expected_subjects = set(expected_subjects_list)
        acc_subjects = set(acc_split_metadata["subject"].astype(str))
        bvp_subjects = set(bvp_split_metadata["subject"].astype(str))
        eda_subjects = set(eda_split_metadata["subject"].astype(str))
        temp_subjects = set(temp_split_metadata["subject"].astype(str))
        split_subjects[split] = acc_subjects

        subjects_match = (
            acc_subjects == expected_subjects == bvp_subjects == eda_subjects == temp_subjects
        )
        bvp_match = metadata_matches_bvp(acc_split_metadata, bvp_split_metadata)
        eda_match = metadata_matches_low_rate_reference(acc_split_metadata, eda_split_metadata)
        temp_match = metadata_matches_low_rate_reference(acc_split_metadata, temp_split_metadata)

        print(f"{split}:")
        print(f"  expected subjects: {sorted(expected_subjects)}")
        print(f"  ACC subjects:      {sorted(acc_subjects)}")
        print(f"  BVP subjects:      {sorted(bvp_subjects)}")
        print(f"  EDA subjects:      {sorted(eda_subjects)}")
        print(f"  TEMP subjects:     {sorted(temp_subjects)}")
        print(f"  subjects match:    {subjects_match}")
        print(f"  BVP metadata match:{bvp_match}")
        print(f"  EDA metadata match:{eda_match}")
        print(f"  TEMP metadata match:{temp_match}")

        if not subjects_match:
            raise ValueError(f"{split}: ACC subjects do not match expected references.")
        if not bvp_match:
            raise ValueError(f"{split}: ACC split metadata does not align with BVP metadata.")
        if not eda_match:
            raise ValueError(f"{split}: ACC split metadata does not align with EDA metadata.")
        if not temp_match:
            raise ValueError(f"{split}: ACC split metadata does not align with TEMP metadata.")

    overlaps = {
        "train_val": sorted(split_subjects["train"] & split_subjects["val"]),
        "train_test": sorted(split_subjects["train"] & split_subjects["test"]),
        "val_test": sorted(split_subjects["val"] & split_subjects["test"]),
    }
    print(f"Subject overlap consistency: {overlaps}")
    if any(overlaps.values()):
        raise ValueError(f"Subject overlap detected: {overlaps}")


def main() -> None:
    args = parse_args()
    acc_dir = args.acc_dir.resolve()

    print("=" * 70)
    print("VITALAGENT - WESAD ACC VERIFICATION")
    print("=" * 70)
    print(f"ACC directory: {acc_dir}")
    print(
        f"Expected ACC window: {WINDOW_SECONDS} seconds / "
        f"{ACC_WINDOW} samples x {ACC_AXES} axes at {ACC_FS} Hz"
    )

    X, y, metadata = load_dataset(acc_dir)
    validate_dataset("FULL ACC", X, y, metadata)
    print_summary("FULL ACC", dataset_summary(X, y, metadata))

    for split, subjects in SPLIT_SUBJECTS.items():
        X_split, y_split, metadata_split = load_dataset(acc_dir, split)
        name = f"{split.upper()} ACC"
        validate_dataset(name, X_split, y_split, metadata_split, set(subjects))
        print_summary(name, dataset_summary(X_split, y_split, metadata_split))

    verify_reference_alignment(acc_dir)

    print("\n" + "=" * 70)
    print("WESAD ACC verification completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
