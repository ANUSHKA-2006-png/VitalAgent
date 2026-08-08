from pathlib import Path

import numpy as np
import torch

from momentfm import MOMENTPipeline


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_rate"
    / "splits"
)


# ============================================================
# LOAD ONE TEST SAMPLE
# ============================================================

X_test = np.load(DATA_PATH / "X_test.npy")
y_test = np.load(DATA_PATH / "y_test.npy")

print("=" * 70)
print("VITALAGENT — MOMENT TEST")
print("=" * 70)

print("\nTest dataset:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# Take only ONE BVP window
sample = X_test[0]

true_hr = y_test[0]

print("\nSingle sample:")
print("Shape:", sample.shape)
print("True HR:", true_hr)


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

# MOMENT expects:
# (batch, channels, sequence_length)

x = torch.tensor(
    sample,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

print("\nInput tensor:")
print("Shape:", x.shape)


# ============================================================
# LOAD MOMENT
# ============================================================

print("\nLoading MOMENT...")

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "embedding"
    },
)

model.init()

model.eval()

print("✅ MOMENT loaded")


# ============================================================
# GENERATE EMBEDDING
# ============================================================

print("\nGenerating embedding...")

with torch.no_grad():

    output = model(x_enc=x)


# ============================================================
# INSPECT OUTPUT
# ============================================================

print("\nOutput type:")
print(type(output))

print("\nOutput:")
print(output)

print("\n" + "=" * 70)
print("✅ MOMENT TEST COMPLETED")
print("=" * 70)