from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# VITALAGENT — CREATING SUBJECT-WISE WESAD SPLIT
# ============================================================

DATA_DIR = Path("data/processed/stress")
SPLIT_DIR = DATA_DIR / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")
metadata = pd.read_csv(DATA_DIR / "metadata.csv")

# Same subject-wise strategy used for the HR pipeline
TRAIN_SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6",
    "S7", "S8", "S9", "S10", "S11"
]

VAL_SUBJECTS = [
    "S13", "S14"
]

TEST_SUBJECTS = [
    "S15", "S16", "S17"
]

print("=" * 70)
print("VITALAGENT — CREATING SUBJECT-WISE WESAD SPLIT")
print("=" * 70)

def create_split(subjects, name):

    mask = metadata["subject"].isin(subjects).values

    X_split = X[mask]
    y_split = y[mask]
    metadata_split = metadata[mask].reset_index(drop=True)

    np.save(SPLIT_DIR / f"X_{name}.npy", X_split)
    np.save(SPLIT_DIR / f"y_{name}.npy", y_split)

    metadata_split.to_csv(
        SPLIT_DIR / f"metadata_{name}.csv",
        index=False
    )

    print(f"\n{name.upper()} SET")
    print("Subjects:", subjects)
    print("X:", X_split.shape)
    print("y:", y_split.shape)

    print("\nClass distribution:")
    print(metadata_split["stress_label"].value_counts().sort_index())

    return set(metadata_split["subject"])


train_subjects = create_split(TRAIN_SUBJECTS, "train")
val_subjects = create_split(VAL_SUBJECTS, "val")
test_subjects = create_split(TEST_SUBJECTS, "test")

# ============================================================
# Verify no subject overlap
# ============================================================

print("\n" + "=" * 70)

print("TRAIN subjects:", sorted(train_subjects))
print("VAL subjects:  ", sorted(val_subjects))
print("TEST subjects: ", sorted(test_subjects))

assert train_subjects.isdisjoint(val_subjects)
assert train_subjects.isdisjoint(test_subjects)
assert val_subjects.isdisjoint(test_subjects)

print("\n✅ No subject overlap")

# ============================================================
# Final summary
# ============================================================

print("\nFiles saved in:")
print(SPLIT_DIR)

print("\n✅ WESAD subject-wise split completed successfully!")