from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_rate"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_rate"
    / "splits"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VITALAGENT — CREATING SUBJECT-WISE HEART-RATE SPLIT")
print("=" * 70)

X = np.load(INPUT_PATH / "X.npy")
y = np.load(INPUT_PATH / "y.npy")

metadata = pd.read_csv(
    INPUT_PATH / "metadata.csv"
)


# ============================================================
# DEFINE SUBJECTS
# ============================================================

TRAIN_SUBJECTS = [
    "S1", "S2", "S3", "S4", "S5",
    "S6", "S7", "S8", "S9", "S10"
]

VAL_SUBJECTS = [
    "S11", "S12"
]

TEST_SUBJECTS = [
    "S13", "S14", "S15"
]


# ============================================================
# CHECK SUBJECTS
# ============================================================

available_subjects = set(
    metadata["subject"].unique()
)

required_subjects = set(
    TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS
)

missing_subjects = required_subjects - available_subjects

if missing_subjects:
    raise ValueError(
        f"Missing subjects: {missing_subjects}"
    )


# ============================================================
# CREATE MASKS
# ============================================================

train_mask = metadata["subject"].isin(
    TRAIN_SUBJECTS
).to_numpy()

val_mask = metadata["subject"].isin(
    VAL_SUBJECTS
).to_numpy()

test_mask = metadata["subject"].isin(
    TEST_SUBJECTS
).to_numpy()


# ============================================================
# SPLIT
# ============================================================

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]


metadata_train = metadata[train_mask].copy()
metadata_val = metadata[val_mask].copy()
metadata_test = metadata[test_mask].copy()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nTRAINING SET")
print("-" * 50)
print("Subjects:", TRAIN_SUBJECTS)
print("X:", X_train.shape)
print("y:", y_train.shape)

print("\nVALIDATION SET")
print("-" * 50)
print("Subjects:", VAL_SUBJECTS)
print("X:", X_val.shape)
print("y:", y_val.shape)

print("\nTEST SET")
print("-" * 50)
print("Subjects:", TEST_SUBJECTS)
print("X:", X_test.shape)
print("y:", y_test.shape)


# ============================================================
# VERIFY NO OVERLAP
# ============================================================

train_set = set(TRAIN_SUBJECTS)
val_set = set(VAL_SUBJECTS)
test_set = set(TEST_SUBJECTS)

if train_set & val_set:
    raise ValueError("Train/validation subject overlap!")

if train_set & test_set:
    raise ValueError("Train/test subject overlap!")

if val_set & test_set:
    raise ValueError("Validation/test subject overlap!")

print("\n✅ No subject overlap")


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_PATH / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_PATH / "y_train.npy",
    y_train
)

np.save(
    OUTPUT_PATH / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_PATH / "y_val.npy",
    y_val
)

np.save(
    OUTPUT_PATH / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_PATH / "y_test.npy",
    y_test
)

metadata_train.to_csv(
    OUTPUT_PATH / "metadata_train.csv",
    index=False
)

metadata_val.to_csv(
    OUTPUT_PATH / "metadata_val.csv",
    index=False
)

metadata_test.to_csv(
    OUTPUT_PATH / "metadata_test.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("✅ SPLIT CREATED SUCCESSFULLY")
print("=" * 70)

print("\nFiles saved in:")
print(OUTPUT_PATH)