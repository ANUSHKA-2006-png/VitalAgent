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
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stress" / "eda"
SPLIT_DIR = OUTPUT_DIR / "splits"

EDA_FS = 4
LABEL_FS = 700
WINDOW_SECONDS = 8

EDA_WINDOW = EDA_FS * WINDOW_SECONDS
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
        description="Preprocess WESAD wrist EDA into 8-second stress windows."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-bvp-alignment-check",
        action="store_true",
        help="Skip compatibility check against existing BVP metadata.csv.",
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
    # Match the original BVP preprocessing order, which used plain Path sorting.
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
            "class_distribution": class_distribution(y_split),
        }

    overlaps = {
        "train_val": sorted(subjects_by_split["train"] & subjects_by_split["val"]),
        "train_test": sorted(subjects_by_split["train"] & subjects_by_split["test"]),
        "val_test": sorted(subjects_by_split["val"] & subjects_by_split["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Subject overlap detected in EDA splits: {overlaps}")
    summary["subject_overlap"] = overlaps
    return summary


def class_distribution(y: Iterable[int] | np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(list(y), dtype=np.int64), minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def check_bvp_metadata_alignment(metadata: pd.DataFrame) -> None:
    if not BVP_METADATA_PATH.exists():
        raise FileNotFoundError(f"Existing BVP metadata not found: {BVP_METADATA_PATH}")

    bvp_metadata = pd.read_csv(BVP_METADATA_PATH)
    compare_columns = [
        "subject",
        "window_id",
        "start_time_seconds",
        "end_time_seconds",
        "wesad_label",
        "stress_label",
    ]

    if len(metadata) != len(bvp_metadata):
        raise ValueError(
            "EDA and existing BVP metadata row counts differ: "
            f"{len(metadata)} != {len(bvp_metadata)}"
        )

    eda_compare = metadata[compare_columns].reset_index(drop=True)
    bvp_compare = bvp_metadata[compare_columns].reset_index(drop=True)
    if not eda_compare.equals(bvp_compare):
        mismatch = np.flatnonzero(
            (eda_compare.astype(str) != bvp_compare.astype(str)).any(axis=1).to_numpy()
        )
        first = int(mismatch[0]) if len(mismatch) else None
        raise ValueError(f"EDA metadata does not align with BVP metadata. First mismatch: {first}")


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VITALAGENT - WESAD EDA PREPROCESSING")
    print("=" * 70)
    print(f"Raw root: {raw_root}")
    print(f"Output: {output_dir}")
    print(f"EDA sampling rate: {EDA_FS} Hz")
    print(f"Window: {WINDOW_SECONDS} seconds / {EDA_WINDOW} samples")
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

        eda = np.asarray(wrist["EDA"]).squeeze().astype(np.float32)
        bvp = np.asarray(wrist["BVP"]).squeeze()
        temp = np.asarray(wrist["TEMP"]).squeeze()
        acc = np.asarray(wrist["ACC"])
        labels = np.asarray(data["label"])

        inferred_eda_fs = infer_sampling_rate(len(eda), len(labels))
        if not np.isclose(inferred_eda_fs, EDA_FS, atol=1e-6):
            raise ValueError(
                f"{subject}: inferred EDA sampling rate {inferred_eda_fs:.6f} "
                f"does not match expected {EDA_FS} Hz."
            )

        num_windows = min(len(eda) // EDA_WINDOW, len(labels) // LABEL_WINDOW)
        accepted_windows = 0

        print(f"Available wrist signals: {sorted(wrist.keys())}")
        print(f"EDA samples: {len(eda)}")
        print(f"Label samples: {len(labels)}")
        print(f"Inferred EDA sampling rate: {inferred_eda_fs:.2f} Hz")

        raw_summary_rows.append(
            {
                "subject": subject,
                "wrist_signals": ",".join(sorted(wrist.keys())),
                "eda_samples": int(len(eda)),
                "bvp_samples": int(len(bvp)),
                "temp_samples": int(len(temp)),
                "acc_samples": int(len(acc)),
                "label_samples": int(len(labels)),
                "eda_sampling_rate_hz": float(inferred_eda_fs),
                "unique_wesad_labels": ",".join(map(str, sorted(np.unique(labels).astype(int)))),
            }
        )

        for window_id in range(num_windows):
            eda_start = window_id * EDA_WINDOW
            eda_end = eda_start + EDA_WINDOW
            label_start = window_id * LABEL_WINDOW
            label_end = label_start + LABEL_WINDOW

            label_window = labels[label_start:label_end]
            wesad_label = dominant_label(label_window)
            stress_label = stress_label_from_wesad_label(wesad_label)
            if stress_label is None:
                continue

            X_all.append(eda[eda_start:eda_end])
            y_all.append(stress_label)
            metadata_rows.append(
                {
                    "subject": subject,
                    "window_id": int(window_id),
                    "start_sample": int(eda_start),
                    "end_sample": int(eda_end),
                    "start_time_seconds": float(eda_start / EDA_FS),
                    "end_time_seconds": float(eda_end / EDA_FS),
                    "label_start_sample": int(label_start),
                    "label_end_sample": int(label_end),
                    "eda_sampling_rate_hz": EDA_FS,
                    "wesad_label": int(wesad_label),
                    "stress_label": int(stress_label),
                }
            )
            accepted_windows += 1

        print(f"Complete 8-second EDA windows: {num_windows}")
        print(f"Accepted windows: {accepted_windows}")

        del data, wrist, eda, bvp, temp, acc, labels
        gc.collect()

    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    metadata = pd.DataFrame(metadata_rows)
    raw_summary = pd.DataFrame(raw_summary_rows)

    if X.ndim != 2 or X.shape[1] != EDA_WINDOW:
        raise ValueError(f"Expected EDA X shape (n, {EDA_WINDOW}), got {X.shape}")
    if len(X) != len(y) or len(y) != len(metadata):
        raise ValueError("EDA X, y, and metadata row counts do not match.")
    if not np.isfinite(X).all():
        raise ValueError("EDA X contains NaN or Inf values.")

    if not args.skip_bvp_alignment_check:
        check_bvp_metadata_alignment(metadata)
        print("\nEDA metadata alignment with existing BVP metadata: OK")

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    raw_summary.to_csv(output_dir / "raw_signal_summary.csv", index=False)
    split_summary = build_splits(X, y, metadata, output_dir)

    print("\n" + "=" * 70)
    print("FINAL EDA DATASET")
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
    print("\nWESAD EDA preprocessing completed successfully.")


if __name__ == "__main__":
    main()
