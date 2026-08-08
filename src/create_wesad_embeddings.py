import numpy as np
import torch
from momentfm import MOMENTPipeline
from pathlib import Path


# ============================================================
# VITALAGENT — WESAD MOMENT EMBEDDINGS
# ============================================================

BASE_DIR = Path("data/processed/stress/splits")
OUTPUT_DIR = Path("data/processed/stress/embeddings")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

print("=" * 60)
print("VITALAGENT — WESAD MOMENT EMBEDDING GENERATION")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")


# ------------------------------------------------------------
# Load MOMENT
# ------------------------------------------------------------

print("\nLoading MOMENT...")

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "embedding",
    },
)

model.init()
model.to(DEVICE)
model.eval()

print("✅ MOMENT loaded")


# ------------------------------------------------------------
# Embedding function
# ------------------------------------------------------------

def generate_embeddings(X, split_name):

    print(f"\nGenerating {split_name} embeddings...")
    print(f"Input shape: {X.shape}")

    embeddings = []

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(X))

            batch = X[start:end]

            # (batch, 512) → (batch, 1, 512)
            batch_tensor = torch.tensor(
                batch,
                dtype=torch.float32
            ).unsqueeze(1)

            batch_tensor = batch_tensor.to(DEVICE)

            output = model(x_enc=batch_tensor)

            batch_embeddings = output.embeddings

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

            print(
                f"Processed {end}/{len(X)}",
                end="\r"
            )

    embeddings = np.concatenate(embeddings, axis=0)

    print()
    print(f"{split_name} embeddings shape: {embeddings.shape}")

    return embeddings


# ------------------------------------------------------------
# Load WESAD splits
# ------------------------------------------------------------

print("\nLoading WESAD splits...")

X_train = np.load(BASE_DIR / "X_train.npy")
X_val = np.load(BASE_DIR / "X_val.npy")
X_test = np.load(BASE_DIR / "X_test.npy")

y_train = np.load(BASE_DIR / "y_train.npy")
y_val = np.load(BASE_DIR / "y_val.npy")
y_test = np.load(BASE_DIR / "y_test.npy")

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# ------------------------------------------------------------
# Generate embeddings
# ------------------------------------------------------------

train_embeddings = generate_embeddings(
    X_train,
    "TRAIN"
)

val_embeddings = generate_embeddings(
    X_val,
    "VALIDATION"
)

test_embeddings = generate_embeddings(
    X_test,
    "TEST"
)


# ------------------------------------------------------------
# Save embeddings
# ------------------------------------------------------------

np.save(
    OUTPUT_DIR / "X_train_embeddings.npy",
    train_embeddings
)

np.save(
    OUTPUT_DIR / "X_val_embeddings.npy",
    val_embeddings
)

np.save(
    OUTPUT_DIR / "X_test_embeddings.npy",
    test_embeddings
)

# Save labels as well for convenience
np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)


# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EMBEDDING GENERATION COMPLETED")
print("=" * 60)

print("Train embeddings:", train_embeddings.shape)
print("Val embeddings:  ", val_embeddings.shape)
print("Test embeddings: ", test_embeddings.shape)

print("\nSaved in:")
print(OUTPUT_DIR)

print("\nExpected:")
print("Train → (3732, 1024)")
print("Val   → (756, 1024)")
print("Test  → (1132, 1024)")

print("\n✅ WESAD MOMENT embedding pipeline completed successfully!")