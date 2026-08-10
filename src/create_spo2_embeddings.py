import os
import numpy as np
import torch
from pathlib import Path
from momentfm import MOMENTPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "spo2" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "moment_embeddings" / "spo2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_embeddings(model, X, device, batch_size=16):
    embeddings = []
    total = len(X)
    print(f"Generating embeddings for {total} samples...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = X[start:end]
        tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(x_enc=tensor)
            batch_emb = output.embeddings.cpu().numpy()

        embeddings.append(batch_emb)

    return np.concatenate(embeddings, axis=0)

def main():
    print("=" * 60)
    print("VITALAGENT — GENERATING MOMENT SpO2 EMBEDDINGS")
    print("=" * 60)

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"}
    )
    model.init()
    model.to(device)
    model.eval()

    train_emb = generate_embeddings(model, X_train, device)
    val_emb = generate_embeddings(model, X_val, device)
    test_emb = generate_embeddings(model, X_test, device)

    np.save(OUTPUT_DIR / "train_embeddings.npy", train_emb)
    np.save(OUTPUT_DIR / "train_y.npy", y_train)

    np.save(OUTPUT_DIR / "val_embeddings.npy", val_emb)
    np.save(OUTPUT_DIR / "val_y.npy", y_val)

    np.save(OUTPUT_DIR / "test_embeddings.npy", test_emb)
    np.save(OUTPUT_DIR / "test_y.npy", y_test)

    print("\n" + "=" * 60)
    print("[OK] SpO2 EMBEDDINGS SAVED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
