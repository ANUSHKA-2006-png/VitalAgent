from pathlib import Path
import pickle

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PPG-DaLiA"
    / "data"
    / "PPG_FieldStudy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_rate"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# PPG-DaLiA SETTINGS
# ============================================================

BVP_FS = 64

WINDOW_SECONDS = 8
SHIFT_SECONDS = 2

WINDOW_SIZE = BVP_FS * WINDOW_SECONDS
SHIFT_SIZE = BVP_FS * SHIFT_SECONDS


# ============================================================
# STORAGE
# ============================================================

all_windows = []
all_labels = []
metadata = []


# ============================================================
# PROCESS EACH SUBJECT
# ============================================================

subject_folders = sorted(
    [
        folder
        for folder in RAW_PATH.iterdir()
        if folder.is_dir() and folder.name.startswith("S")
    ],
    key=lambda x: int(x.name[1:])
)


print("=" * 70)
print("VITALAGENT — PPG-DaLiA HEART-RATE PREPROCESSING")
print("=" * 70)

print(f"\nFound subjects: {len(subject_folders)}")

for subject_folder in subject_folders:

    subject = subject_folder.name

    pkl_file = subject_folder / f"{subject}.pkl"

    if not pkl_file.exists():
        print(f"\n⚠️ {subject}: {pkl_file.name} not found")
        continue

    print(f"\nProcessing {subject}...")

    # --------------------------------------------------------
    # Load subject data
    # --------------------------------------------------------

    with open(pkl_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # --------------------------------------------------------
    # Extract BVP and labels
    # --------------------------------------------------------

    bvp = np.asarray(
        data["signal"]["wrist"]["BVP"]
    ).squeeze()

    labels = np.asarray(
        data["label"]
    ).squeeze()

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    subject_windows = []
    subject_labels = []

    start = 0
    window_id = 0

    while start + WINDOW_SIZE <= len(bvp):

        end = start + WINDOW_SIZE

        window = bvp[start:end]

        subject_windows.append(window)
        subject_labels.append(labels[window_id])

        metadata.append(
            {
                "subject": subject,
                "window_id": window_id,
                "start_sample": start,
                "end_sample": end,
                "start_time_seconds": start / BVP_FS,
                "end_time_seconds": end / BVP_FS,
                "heart_rate_bpm": float(labels[window_id]),
            }
        )

        start += SHIFT_SIZE
        window_id += 1

    subject_windows = np.asarray(subject_windows)
    subject_labels = np.asarray(subject_labels)

    # --------------------------------------------------------
    # Verify alignment
    # --------------------------------------------------------

    if len(subject_windows) != len(subject_labels):
        raise ValueError(
            f"{subject}: number of windows and labels do not match"
        )

    print(
        f"   Windows: {subject_windows.shape}"
    )

    print(
        f"   Labels:  {subject_labels.shape}"
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    all_windows.extend(subject_windows)
    all_labels.extend(subject_labels)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.asarray(all_windows, dtype=np.float32)
y = np.asarray(all_labels, dtype=np.float32)

metadata_df = pd.DataFrame(metadata)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("metadata shape:", metadata_df.shape)

if len(X) != len(y):
    raise ValueError("X and y have different numbers of samples!")

if len(X) != len(metadata_df):
    raise ValueError("X and metadata have different numbers of samples!")


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_PATH / "X.npy",
    X
)

np.save(
    OUTPUT_PATH / "y.npy",
    y
)

metadata_df.to_csv(
    OUTPUT_PATH / "metadata.csv",
    index=False
)


print("\n✅ Saved:")
print("   ", OUTPUT_PATH / "X.npy")
print("   ", OUTPUT_PATH / "y.npy")
print("   ", OUTPUT_PATH / "metadata.csv")

print("\n✅ PPG-DaLiA preprocessing completed successfully!")