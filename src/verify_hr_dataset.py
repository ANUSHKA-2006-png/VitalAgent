from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_rate"
)


# ============================================================
# LOAD
# ============================================================

X = np.load(DATA_PATH / "X.npy")
y = np.load(DATA_PATH / "y.npy")
metadata = pd.read_csv(DATA_PATH / "metadata.csv")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 70)
print("VITALAGENT — HEART-RATE DATASET VERIFICATION")
print("=" * 70)

print("\nX:")
print("  Shape:", X.shape)
print("  Data type:", X.dtype)

print("\ny:")
print("  Shape:", y.shape)
print("  Data type:", y.dtype)

print("\nMetadata:")
print("  Shape:", metadata.shape)


# ============================================================
# CHECK SAMPLE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE COUNT CHECK")
print("=" * 70)

print("X samples:", len(X))
print("y samples:", len(y))
print("metadata rows:", len(metadata))

if len(X) == len(y) == len(metadata):
    print("✅ All sample counts match")
else:
    print("❌ Sample counts do NOT match")


# ============================================================
# CHECK MISSING / INVALID VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print("NaN values in X:", np.isnan(X).sum())
print("NaN values in y:", np.isnan(y).sum())

print("Infinite values in X:", np.isinf(X).sum())
print("Infinite values in y:", np.isinf(y).sum())


# ============================================================
# HR STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("HEART-RATE STATISTICS")
print("=" * 70)

print(f"Minimum HR: {y.min():.2f} BPM")
print(f"Maximum HR: {y.max():.2f} BPM")
print(f"Mean HR:    {y.mean():.2f} BPM")
print(f"Median HR:  {np.median(y):.2f} BPM")
print(f"Std HR:     {y.std():.2f} BPM")


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT DISTRIBUTION")
print("=" * 70)

subject_counts = metadata["subject"].value_counts().sort_index()

print(subject_counts)


# ============================================================
# CHECK METADATA
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 METADATA ROWS")
print("=" * 70)

print(metadata.head())


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

if (
    len(X) == len(y) == len(metadata)
    and not np.isnan(X).any()
    and not np.isnan(y).any()
    and not np.isinf(X).any()
    and not np.isinf(y).any()
):
    print("✅ DATASET VERIFICATION PASSED")
else:
    print("⚠️ DATASET NEEDS ATTENTION")

print("=" * 70)