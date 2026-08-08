import numpy as np
from pathlib import Path

BASE = Path("data/processed/stress/embeddings")

print("=" * 60)
print("VITALAGENT — WESAD EMBEDDING VERIFICATION")
print("=" * 60)

for split in ["train", "val", "test"]:
    X = np.load(BASE / f"X_{split}_embeddings.npy")
    y = np.load(BASE / f"y_{split}.npy")

    print(f"\n{split.upper()}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)
    print("NaN:", np.isnan(X).sum())
    print("Inf:", np.isinf(X).sum())

    print("Class distribution:")
    unique, counts = np.unique(y, return_counts=True)

    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count}")

    assert len(X) == len(y)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

print("\n✅ WESAD embeddings verified successfully!")