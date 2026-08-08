from pathlib import Path
import numpy as np
import pandas as pd

X_PATH = Path("data/processed/stress/X.npy")
Y_PATH = Path("data/processed/stress/y.npy")
META_PATH = Path("data/processed/stress/metadata.csv")

print("=" * 70)
print("VITALAGENT — WESAD DATASET VERIFICATION")
print("=" * 70)

X = np.load(X_PATH)
y = np.load(Y_PATH)
metadata = pd.read_csv(META_PATH)

print("\nShapes:")
print("X:", X.shape)
print("y:", y.shape)
print("Metadata:", metadata.shape)

print("\nData types:")
print("X:", X.dtype)
print("y:", y.dtype)

print("\nSample counts:")
print("X samples:", len(X))
print("y samples:", len(y))
print("metadata rows:", len(metadata))

assert len(X) == len(y) == len(metadata)

print("✅ All sample counts match")

print("\nMissing values:")
print("NaN in X:", np.isnan(X).sum())
print("NaN in y:", np.isnan(y).sum())

print("\nInfinite values:")
print("Inf in X:", np.isinf(X).sum())
print("Inf in y:", np.isinf(y).sum())

print("\nBVP statistics:")
print("Minimum:", X.min())
print("Maximum:", X.max())
print("Mean:", X.mean())
print("Std:", X.std())

print("\nClass distribution:")
print(metadata["stress_label"].value_counts().sort_index())

print("\nClass percentages:")
print(
    metadata["stress_label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nSubject distribution:")
print(metadata["subject"].value_counts().sort_index())

print("\nUnique WESAD labels:")
print(sorted(metadata["wesad_label"].unique()))

print("\nFirst 5 metadata rows:")
print(metadata.head())

print("\n" + "=" * 70)
print("✅ WESAD VERIFICATION COMPLETED")
print("=" * 70)