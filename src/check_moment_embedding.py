import numpy as np
import torch
from momentfm import MOMENTPipeline

X_test = np.load(
    "data/processed/heart_rate/splits/X_test.npy"
)

sample = torch.tensor(
    X_test[0],
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

print("Input shape:", sample.shape)

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

with torch.no_grad():
    output = model(x_enc=sample)

print("\nOutput type:")
print(type(output))

print("\nEmbedding shape:")
print(output.embeddings.shape)

print("\nEmbedding dtype:")
print(output.embeddings.dtype)