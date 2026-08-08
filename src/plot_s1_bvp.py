from pathlib import Path
import pickle

import matplotlib.pyplot as plt
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

BVP_FS = 64
WINDOW_SECONDS = 8
SHIFT_SECONDS = 2

WINDOW_SIZE = BVP_FS * WINDOW_SECONDS
SHIFT_SIZE = BVP_FS * SHIFT_SECONDS


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

with open(PKL_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

bvp = np.asarray(
    data["signal"]["wrist"]["BVP"]
).squeeze()

labels = np.asarray(data["label"])


# ---------------------------------------------------------
# Create first window
# ---------------------------------------------------------

first_window = bvp[:WINDOW_SIZE]

time = np.arange(WINDOW_SIZE) / BVP_FS


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(time, first_window)

plt.xlabel("Time (seconds)")
plt.ylabel("BVP amplitude")

plt.title(
    f"S1 PPG-DaLiA — First 8-Second BVP Window\n"
    f"Ground Truth HR: {labels[0]:.2f} BPM"
)

plt.grid(True)

plt.tight_layout()

plt.show()