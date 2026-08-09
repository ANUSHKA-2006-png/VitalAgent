from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "WESAD" / "WESAD"
BVP_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "stress" / "metadata.csv"
EDA_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "stress" / "eda" / "metadata.csv"
TEMP_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "stress" / "temp" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stress" / "acc"

ACC_FS = 32
ACC_AXES = 3
LABEL_FS = 700
WINDOW_SECONDS = 8

ACC_WINDOW = ACC_FS * WINDOW_SECONDS
LABEL_WINDOW = LABEL_FS * WINDOW_SECONDS

STRESS_LABEL = 2
NON_STRESS_LABELS = {1, 3, 4}
IGNORE_LABELS = {0, 5, 6, 7}

TRAIN_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"]
VAL_SUBJECTS = ["S13", "S14"]
TEST_SUBJECTS = ["S15", "S16", "S17"]
SPLIT_SUBJECTS = {
    "train": TRAIN_SUBJECTS,
    "val": VAL_SUBJECTS,
    "test": TEST_SUBJECTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess WESAD wrist ACC into 8-second stress windows."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-reference-alignment-check",
        action="store_true",
        help="Skip compatibility checks against existing BVP, EDA, and TEMP metadata.",
    )
    return parser.parse_args()


def stress_label_from_wesad_label(wesad_label: int) -> int | None:
    if wesad_label == STRESS_LABEL:
        return 1
    if wesad_label in NON_STRESS_LABELS:
        return 0
    if wesad_label in IGNORE_LABELS:
        return None
    return None


def dominant_label(label_window: np.ndarray) -> int:
    unique, counts = np.unique(label_window, return_counts=True)
    return int(unique[np.argmax(counts)])


def infer_sampling_rate(signal_length: int, label_length: int) -> float:
    duration_seconds = label_length / LABEL_FS
    return signal_length / duration_seconds


def iter_subject_files(raw_root: Path) -> list[Path]:
    # Match the original BVP/EDA/TEMP preprocessing order, which used plain Path sorting.
    subject_files = sorted(raw_root.glob("S*/S*.pkl"))
    if not subject_files:
        raise FileNotFoundError(f"No WESAD subject pickles found under {raw_root}")
    return subject_files


def validate_wrist_signal_keys(subject: str, wrist: dict) -> None:
    available = set(wrist)
    required = {"ACC", "BVP", "EDA", "TEMP"}
    missing = required - available
    if missing:
        raise ValueError(f"{subject}: missing wrist signal(s): {sorted(missing)}")


def class_distribution(y: Iterable[int] | np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def metadata_matches_bvp(acc_metadata: pd.DataFrame, bvp_metadata: pd.DataFrame) -> bool:
    exact_columns = ["subject", "window_id", "wesad_label", "stress_label"]
    time_columns = ["start_time_seconds", "end_time_seconds"]

    if len(acc_metadata) != len(bvp_metadata):
        return False
    for column in exact_columns:
        if not acc_metadata[column].reset_index(drop=True).equals(
            bvp_metadata[column].reset_index(drop=True)
        ):
            return False
    for column in time_columns:
        if not np.allclose(
            acc_metadata[column].to_numpy(dtype=float),
            bvp_metadata[column].to_numpy(dtype=float),
        ):
            return False
    return True


def metadata_matches_low_rate_reference(
    acc_metadata: pd.DataFrame,
    reference_metadata: pd.DataFrame,
) -> bool:
    exact_columns = [
        "subject",
        "window_id",
        "label_start_sample",
        "label_end_sample",
        "wesad_label",
        "stress_label",
    ]
    time_columns = ["start_time_seconds", "end_time_seconds"]

    if len(acc_metadata) != len(reference_metadata):
        return False
    for column in exact_columns:
        if not acc_metadata[column].reset_index(drop=True).equals(
            reference_metadata[column].reset_index(drop=True)
        ):
            return False
    for column in time_columns:
        if not np.allclose(
            acc_metadata[column].to_numpy(dtype=float),
            reference_metadata[column].to_numpy(dtype=float),
        ):
            return False

    # EDA/TEMP sample indices are 4 Hz while ACC is 32 Hz, so compare sample
    # identity after mapping both to seconds.
    acc_start_seconds = acc_metadata["start_sample"].to_numpy(dtype=float) / ACC_FS
    acc_end_seconds = acc_metadata["end_sample"].to_numpy(dtype=float) / ACC_FS
    reference_rate_column = [
        column
        for column in ("eda_sampling_rate_hz", "temp_sampling_rate_hz")
        if column in reference_metadata.columns
    ][0]
    reference_fs = reference_metadata[reference_rate_column].to_numpy(dtype=float)
    reference_start_seconds = reference_metadata["start_sample"].to_numpy(dtype=float) / reference_fs
    reference_end_seconds = reference_metadata["end_sample"].to_numpy(dtype=float) / reference_fs

    return bool(
        np.allclose(acc_start_seconds, reference_start_seconds)
        and np.allclose(acc_end_seconds, reference_end_seconds)
    )


def check_reference_alignment(metadata: pd.DataFrame) -> None:
    for path, name in (
        (BVP_METADATA_PATH, "BVP"),
        (EDA_METADATA_PATH, "EDA"),
        (TEMP_METADATA_PATH, "TEMP"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Existing {name} metadata not found: {path}")

    bvp_metadata = pd.read_csv(BVP_METADATA_PATH)
    eda_metadata = pd.read_csv(EDA_METADATA_PATH)
    temp_metadata = pd.read_csv(TEMP_METADATA_PATH)

    if not metadata_matches_bvp(metadata, bvp_metadata):
        raise ValueError("ACC metadata does not align with existing BVP metadata.")
    if not metadata_matches_low_rate_reference(metadata, eda_metadata):
        raise ValueError("ACC metadata does not align with existing EDA metadata.")
    if not metadata_matches_low_rate_reference(metadata, temp_metadata):
        raise ValueError("ACC metadata does not align with existing TEMP metadata.")


def build_splits(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, object]]:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}
    subjects_by_split: dict[str, set[str]] = {}

    for split, subjects in SPLIT_SUBJECTS.items():
        mask = metadata["subject"].isin(subjects).to_numpy()
        X_split = X[mask]
        y_split = y[mask]
        metadata_split = metadata.loc[mask].reset_index(drop=True)

        np.save(split_dir / f"X_{split}.npy", X_split)
        np.save(split_dir / f"y_{split}.npy", y_split)
        metadata_split.to_csv(split_dir / f"metadata_{split}.csv", index=False)

        actual_subjects = set(metadata_split["subject"].astype(str))
        subjects_by_split[split] = actual_subjects
        summary[split] = {
            "subjects": sorted(actual_subjects),
            "X_shape": list(X_split.shape),
            "y_shape": list(y_split.shape),
            "metadata_shape": list(metadata_split.shape),
            "class_distribution": class_distribution(y_split),
        }

    overlaps = {
        "train_val": sorted(subjects_by_split["train"] & subjects_by_split["val"]),
        "train_test": sorted(subjects_by_split["train"] & subjects_by_split["test"]),
        "val_test": sorted(subjects_by_split["val"] & subjects_by_split["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Subject overlap detected in ACC splits: {overlaps}")
    summary["subject_overlap"] = overlaps
    return summary


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VITALAGENT - WESAD ACC PREPROCESSING")
    print("=" * 70)
    print(f"Raw root: {raw_root}")
    print(f"Output: {output_dir}")
    print(f"ACC sampling rate: {ACC_FS} Hz")
    print(f"ACC axes: {ACC_AXES}")
    print(f"Window: {WINDOW_SECONDS} seconds / {ACC_WINDOW} samples x {ACC_AXES} axes")
    print(f"Label window: {LABEL_WINDOW} samples at {LABEL_FS} Hz")
    print(f"Stress label mapping: WESAD {STRESS_LABEL} -> 1")
    print(f"Non-stress label mapping: WESAD {sorted(NON_STRESS_LABELS)} -> 0")
    print(f"Ignored WESAD labels: {sorted(IGNORE_LABELS)}")

    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    metadata_rows: list[dict[str, object]] = []
    raw_summary_rows: list[dict[str, object]] = []

    subject_files = iter_subject_files(raw_root)
    print(f"\nFound subject pickles: {len(subject_files)}")

    for pkl_path in subject_files:
        subject = pkl_path.parent.name
        print(f"\nProcessing {subject}...")

        with pkl_path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")

        wrist = data["signal"]["wrist"]
        validate_wrist_signal_keys(subject, wrist)

        acc = np.asarray(wrist["ACC"], dtype=np.float32)
        bvp = np.asarray(wrist["BVP"]).squeeze()
        eda = np.asarray(wrist["EDA"]).squeeze()
        temp = np.asarray(wrist["TEMP"]).squeeze()
        labels = np.asarray(data["label"])

        if acc.ndim != 2 or acc.shape[1] != ACC_AXES:
            raise ValueError(f"{subject}: expected ACC shape (n, {ACC_AXES}), got {acc.shape}")

        inferred_acc_fs = infer_sampling_rate(len(acc), len(labels))
        if not np.isclose(inferred_acc_fs, ACC_FS, atol=1e-6):
            raise ValueError(
                f"{subject}: inferred ACC sampling rate {inferred_acc_fs:.6f} "
                f"does not match expected {ACC_FS} Hz."
            )

        num_windows = min(len(acc) // ACC_WINDOW, len(labels) // LABEL_WINDOW)
        accepted_windows = 0

        axis_minimums = np.nanmin(acc, axis=0)
        axis_maximums = np.nanmax(acc, axis=0)

        print(f"Available wrist signals: {sorted(wrist.keys())}")
        print(f"ACC shape: {acc.shape}")
        print(f"Label samples: {len(labels)}")
        print(f"Inferred ACC sampling rate: {inferred_acc_fs:.2f} Hz")

        raw_summary_rows.append(
            {
                "subject": subject,
                "wrist_signals": ",".join(sorted(wrist.keys())),
                "acc_shape": f"{acc.shape[0]}x{acc.shape[1]}",
                "acc_samples": int(len(acc)),
                "acc_axes": int(acc.shape[1]),
                "bvp_samples": int(len(bvp)),
                "eda_samples": int(len(eda)),
                "temp_samples": int(len(temp)),
                "label_samples": int(len(labels)),
                "acc_sampling_rate_hz": float(inferred_acc_fs),
                "acc_axis_minimums": ",".join(map(str, axis_minimums.tolist())),
                "acc_axis_maximums": ",".join(map(str, axis_maximums.tolist())),
                "unique_wesad_labels": ",".join(
                    map(str, sorted(np.unique(labels).astype(int)))
                ),
            }
        )

        for window_id in range(num_windows):
            acc_start = window_id * ACC_WINDOW
            acc_end = acc_start + ACC_WINDOW
            label_start = window_id * LABEL_WINDOW
            label_end = label_start + LABEL_WINDOW

            label_window = labels[label_start:label_end]
            wesad_label = dominant_label(label_window)
            stress_label = stress_label_from_wesad_label(wesad_label)
            if stress_label is None:
                continue

            X_all.append(acc[acc_start:acc_end, :])
            y_all.append(stress_label)
            metadata_rows.append(
                {
                    "subject": subject,
                    "window_id": int(window_id),
                    "start_sample": int(acc_start),
                    "end_sample": int(acc_end),
                    "start_time_seconds": float(acc_start / ACC_FS),
                    "end_time_seconds": float(acc_end / ACC_FS),
                    "label_start_sample": int(label_start),
                    "label_end_sample": int(label_end),
                    "acc_sampling_rate_hz": ACC_FS,
                    "acc_axes": ACC_AXES,
                    "wesad_label": int(wesad_label),
                    "stress_label": int(stress_label),
                }
            )
            accepted_windows += 1

        print(f"Complete 8-second ACC windows: {num_windows}")
        print(f"Accepted windows: {accepted_windows}")

        del data, wrist, acc, bvp, eda, temp, labels
        gc.collect()

    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    metadata = pd.DataFrame(metadata_rows)
    raw_summary = pd.DataFrame(raw_summary_rows)

    expected_shape_tail = (ACC_WINDOW, ACC_AXES)
    if X.ndim != 3 or X.shape[1:] != expected_shape_tail:
        raise ValueError(f"Expected ACC X shape (n, {ACC_WINDOW}, {ACC_AXES}), got {X.shape}")
    if len(X) != len(y) or len(y) != len(metadata):
        raise ValueError("ACC X, y, and metadata row counts do not match.")
    if not np.isfinite(X).all():
        raise ValueError("ACC X contains NaN or Inf values.")

    if not args.skip_reference_alignment_check:
        check_reference_alignment(metadata)
        print("\nACC metadata alignment with existing BVP, EDA, and TEMP metadata: OK")

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    raw_summary.to_csv(output_dir / "raw_signal_summary.csv", index=False)
    split_summary = build_splits(X, y, metadata, output_dir)

    print("\n" + "=" * 70)
    print("FINAL ACC DATASET")
    print("=" * 70)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"metadata shape: {metadata.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")
    print(f"Class distribution: {class_distribution(y)}")
    print("\nSubject distribution:")
    print(metadata["subject"].value_counts().sort_index())

    print("\nSplit summary:")
    for split in ("train", "val", "test"):
        print(f"{split}: {split_summary[split]}")
    print(f"subject_overlap: {split_summary['subject_overlap']}")

    print("\nSaved:")
    print(output_dir / "X.npy")
    print(output_dir / "y.npy")
    print(output_dir / "metadata.csv")
    print(output_dir / "raw_signal_summary.csv")
    print(output_dir / "splits")
    print("\nWESAD ACC preprocessing completed successfully.")


if __name__ == "__main__":
    main()
