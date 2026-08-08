import os
import numpy as np

print("=" * 60)
print("VITALAGENT — VERIFYING MOMENT HR EMBEDDINGS")
print("=" * 60)

BASE = "data/processed/moment_embeddings/heart_rate"

files = [
    "train_embeddings.npy",
    "val_embeddings.npy",
    "test_embeddings.npy",
    "train_y.npy",
    "val_y.npy",
    "test_y.npy",
]

print("\nChecking files...\n")

for file in files:
    path = os.path.join(BASE, file)

    if os.path.exists(path):
        data = np.load(path)

        print(f"✅ {file}")
        print(f"   Shape: {data.shape}")
        print(f"   Type:  {data.dtype}")
    else:
        print(f"❌ Missing: {file}")


print("\n" + "=" * 60)

train_X = np.load(
    os.path.join(BASE, "train_embeddings.npy")
)

val_X = np.load(
    os.path.join(BASE, "val_embeddings.npy")
)

test_X = np.load(
    os.path.join(BASE, "test_embeddings.npy")
)

train_y = np.load(
    os.path.join(BASE, "train_y.npy")
)

val_y = np.load(
    os.path.join(BASE, "val_y.npy")
)

test_y = np.load(
    os.path.join(BASE, "test_y.npy")
)


print("TRAIN")
print("X:", train_X.shape)
print("y:", train_y.shape)

print("\nVALIDATION")
print("X:", val_X.shape)
print("y:", val_y.shape)

print("\nTEST")
print("X:", test_X.shape)
print("y:", test_y.shape)


print("\nNaN check:")
print("Train X:", np.isnan(train_X).sum())
print("Val X:  ", np.isnan(val_X).sum())
print("Test X: ", np.isnan(test_X).sum())


print("\n" + "=" * 60)
print("✅ EMBEDDING VERIFICATION COMPLETED")
print("=" * 60)