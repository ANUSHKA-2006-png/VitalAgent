import os
import numpy as np
import torch
from momentfm import MOMENTPipeline


print("=" * 60)
print("VITALAGENT — GENERATING MOMENT HR EMBEDDINGS")
print("=" * 60)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = "data/processed/heart_rate/splits"

OUTPUT_DIR = "data/processed/moment_embeddings/heart_rate"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("\nLoading datasets...")


X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)


X_val = np.load(
    os.path.join(DATA_DIR, "X_val.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "y_val.npy")
)


X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)


print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# ---------------------------------------------------------
# Load MOMENT
# ---------------------------------------------------------

print("\nLoading MOMENT...")

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "embedding"
    }
)

model.init()
model.eval()

print("✅ MOMENT loaded")


# ---------------------------------------------------------
# Embedding function
# ---------------------------------------------------------

def generate_embeddings(X, batch_size=8):

    embeddings = []

    total = len(X)

    print(f"\nGenerating embeddings for {total} samples...")

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        batch = X[start:end]

        tensor = torch.tensor(
            batch,
            dtype=torch.float32
        ).unsqueeze(1)

        with torch.no_grad():

            output = model(
                x_enc=tensor
            )

        batch_embeddings = output.embeddings

        embeddings.append(
            batch_embeddings.cpu().numpy()
        )

        print(
            f"Processed {end}/{total}"
        )

    return np.concatenate(
        embeddings,
        axis=0
    )


# ---------------------------------------------------------
# Generate TRAIN embeddings
# ---------------------------------------------------------

train_embeddings = generate_embeddings(
    X_train
)

print(
    "\nTrain embeddings:",
    train_embeddings.shape
)


# ---------------------------------------------------------
# Generate VALIDATION embeddings
# ---------------------------------------------------------

val_embeddings = generate_embeddings(
    X_val
)

print(
    "\nValidation embeddings:",
    val_embeddings.shape
)


# ---------------------------------------------------------
# Generate TEST embeddings
# ---------------------------------------------------------

test_embeddings = generate_embeddings(
    X_test
)

print(
    "\nTest embeddings:",
    test_embeddings.shape
)


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

print("\nSaving embeddings...")


np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_embeddings.npy"
    ),
    train_embeddings
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_embeddings.npy"
    ),
    val_embeddings
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_embeddings.npy"
    ),
    test_embeddings
)


# ---------------------------------------------------------
# Save labels
# ---------------------------------------------------------

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_y.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_y.npy"
    ),
    y_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_y.npy"
    ),
    y_test
)


print("\n" + "=" * 60)
print("✅ MOMENT EMBEDDINGS SAVED")
print("=" * 60)

print(
    "\nOutput folder:",
    OUTPUT_DIR
)

print(
    "\nTrain:",
    train_embeddings.shape
)

print(
    "Validation:",
    val_embeddings.shape
)

print(
    "Test:",
    test_embeddings.shape
)