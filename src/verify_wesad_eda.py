from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_wesad_eda import (
    EDA_FS,
    EDA_WINDOW,
    NON_STRESS_LABELS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SPLIT_SUBJECTS,
    STRESS_LABEL,
    WINDOW_SECONDS,
)


BVP_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "stress"
BVP_SPLIT_DIR = BVP_PROCESSED_DIR / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify processed WESAD wrist EDA windows.")
    parser.add_argument("--eda-dir", type=Path, default=OUTPUT_DIR)
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


def dataset_summary(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    return {
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
        "class_distribution": class_distribution(y),
        "subject_distribution": subject_distribution(metadata),
        "unique_original_wesad_labels": sorted(
            metadata["wesad_label"].astype(int).unique().tolist()
        ),
    }


def print_summary(name: str, summary: dict[str, Any]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
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
    print(f"Class distribution: {summary['class_distribution']}")
    print(f"Subject distribution: {summary['subject_distribution']}")
    print(f"Unique original WESAD labels: {summary['unique_original_wesad_labels']}")


def validate_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    expected_subjects: set[str] | None = None,
) -> None:
    if X.ndim != 2 or X.shape[1] != EDA_WINDOW:
        raise ValueError(f"{name}: expected X shape (n, {EDA_WINDOW}), got {X.shape}")
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
    if not np.all(sample_lengths.to_numpy(dtype=int) == EDA_WINDOW):
        raise ValueError(f"{name}: not all EDA windows contain {EDA_WINDOW} samples.")
    if "eda_sampling_rate_hz" in metadata.columns:
        if not np.all(metadata["eda_sampling_rate_hz"].to_numpy(dtype=float) == EDA_FS):
            raise ValueError(f"{name}: unexpected EDA sampling-rate metadata.")


def metadata_matches_reference(
    eda_metadata: pd.DataFrame,
    reference_metadata: pd.DataFrame,
) -> bool:
    exact_columns = ["subject", "window_id", "wesad_label", "stress_label"]
    time_columns = ["start_time_seconds", "end_time_seconds"]

    if len(eda_metadata) != len(reference_metadata):
        return False
    for column in exact_columns:
        if not eda_metadata[column].reset_index(drop=True).equals(
            reference_metadata[column].reset_index(drop=True)
        ):
            return False
    for column in time_columns:
        if not np.allclose(
            eda_metadata[column].to_numpy(dtype=float),
            reference_metadata[column].to_numpy(dtype=float),
        ):
            return False
    return True


def verify_reference_alignment(eda_dir: Path) -> None:
    X_full, y_full, eda_metadata = load_dataset(eda_dir)
    del X_full, y_full
    bvp_metadata = pd.read_csv(BVP_PROCESSED_DIR / "metadata.csv")
    if not metadata_matches_reference(eda_metadata, bvp_metadata):
        raise ValueError("Full EDA metadata does not align with existing BVP metadata.")

    print("\nReference alignment")
    print("-------------------")
    print("Full EDA metadata aligns with existing BVP metadata: True")

    split_subjects: dict[str, set[str]] = {}
    for split, expected_subjects_list in SPLIT_SUBJECTS.items():
        _, _, eda_split_metadata = load_dataset(eda_dir, split)
        bvp_split_metadata = pd.read_csv(BVP_SPLIT_DIR / f"metadata_{split}.csv")

        expected_subjects = set(expected_subjects_list)
        eda_subjects = set(eda_split_metadata["subject"].astype(str))
        bvp_subjects = set(bvp_split_metadata["subject"].astype(str))
        split_subjects[split] = eda_subjects

        metadata_match = metadata_matches_reference(eda_split_metadata, bvp_split_metadata)
        subjects_match = eda_subjects == expected_subjects == bvp_subjects

        print(f"{split}:")
        print(f"  expected subjects: {sorted(expected_subjects)}")
        print(f"  EDA subjects:      {sorted(eda_subjects)}")
        print(f"  BVP subjects:      {sorted(bvp_subjects)}")
        print(f"  subjects match:    {subjects_match}")
        print(f"  metadata match:    {metadata_match}")

        if not subjects_match:
            raise ValueError(f"{split}: EDA subjects do not match expected/BVP subjects.")
        if not metadata_match:
            raise ValueError(f"{split}: EDA split metadata does not align with BVP split metadata.")

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
    eda_dir = args.eda_dir.resolve()

    print("=" * 70)
    print("VITALAGENT - WESAD EDA VERIFICATION")
    print("=" * 70)
    print(f"EDA directory: {eda_dir}")
    print(f"Expected EDA window: {WINDOW_SECONDS} seconds / {EDA_WINDOW} samples at {EDA_FS} Hz")

    X, y, metadata = load_dataset(eda_dir)
    validate_dataset("FULL EDA", X, y, metadata)
    print_summary("FULL EDA", dataset_summary(X, y, metadata))

    for split, subjects in SPLIT_SUBJECTS.items():
        X_split, y_split, metadata_split = load_dataset(eda_dir, split)
        name = f"{split.upper()} EDA"
        validate_dataset(name, X_split, y_split, metadata_split, set(subjects))
        print_summary(name, dataset_summary(X_split, y_split, metadata_split))

    verify_reference_alignment(eda_dir)

    print("\n" + "=" * 70)
    print("WESAD EDA verification completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
