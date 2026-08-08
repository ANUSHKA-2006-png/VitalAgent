from pathlib import Path
import pickle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PKL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PPG-DaLiA"
    / "data"
    / "PPG_FieldStudy"
    / "S1"
    / "S1.pkl"
)

# Dataset information from PPG-DaLiA README
BVP_FS = 64
WINDOW_SECONDS = 8
SHIFT_SECONDS = 2

WINDOW_SIZE = BVP_FS * WINDOW_SECONDS
SHIFT_SIZE = BVP_FS * SHIFT_SECONDS

print("=" * 70)
print("PPG-DaLiA — S1 HEART-RATE WINDOW CHECK")
print("=" * 70)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

with open(PKL_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

bvp = data["signal"]["wrist"]["BVP"]
labels = data["label"]

# Convert BVP to 1D
bvp = np.asarray(bvp).squeeze()

print("\nBVP shape:", bvp.shape)
print("Number of HR labels:", len(labels))

print("\nSampling rate:", BVP_FS, "Hz")
print("Window:", WINDOW_SECONDS, "seconds")
print("Window size:", WINDOW_SIZE, "samples")
print("Shift:", SHIFT_SECONDS, "seconds")
print("Shift size:", SHIFT_SIZE, "samples")

# ---------------------------------------------------------
# Create windows
# ---------------------------------------------------------

windows = []

start = 0

while start + WINDOW_SIZE <= len(bvp):

    end = start + WINDOW_SIZE

    window = bvp[start:end]

    windows.append(window)

    start += SHIFT_SIZE

windows = np.asarray(windows)

print("\nGenerated BVP windows:", windows.shape)

print("\nFirst window:")
print("Shape:", windows[0].shape)

print("\nFirst 10 BVP values:")
print(windows[0][:10])

print("\nFirst 10 HR labels:")
print(labels[:10])