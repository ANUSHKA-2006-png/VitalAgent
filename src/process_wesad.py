from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# ============================================================
# VITALAGENT — WESAD PREPROCESSING
# ============================================================

ROOT = Path("data/raw/WESAD/WESAD")
OUTPUT = Path("data/processed/stress")

OUTPUT.mkdir(parents=True, exist_ok=True)

BVP_FS = 64
LABEL_FS = 700

WINDOW_SECONDS = 8

BVP_WINDOW = BVP_FS * WINDOW_SECONDS
LABEL_WINDOW = LABEL_FS * WINDOW_SECONDS

STRESS_LABEL = 2
NON_STRESS_LABELS = {1, 3, 4}
IGNORE_LABELS = {0, 5, 6, 7}

print("=" * 70)
print("VITALAGENT — WESAD STRESS PREPROCESSING")
print("=" * 70)

X_all = []
y_all = []
metadata = []

subject_files = sorted(ROOT.glob("S*/S*.pkl"))

print(f"\nFound subjects: {len(subject_files)}")

for pkl_path in subject_files:

    subject = pkl_path.parent.name

    print(f"\nProcessing {subject}...")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # --------------------------------------------------------
    # Get wrist BVP
    # --------------------------------------------------------

    bvp = data["signal"]["wrist"]["BVP"].squeeze().astype(np.float32)

    # --------------------------------------------------------
    # Get labels
    # --------------------------------------------------------

    labels = data["label"]

    print(f"BVP samples: {len(bvp)}")
    print(f"Label samples: {len(labels)}")

    # Number of complete 8-second windows
    num_windows = min(
        len(bvp) // BVP_WINDOW,
        len(labels) // LABEL_WINDOW
    )

    subject_count = 0

    for window_id in range(num_windows):

        # BVP window
        bvp_start = window_id * BVP_WINDOW
        bvp_end = bvp_start + BVP_WINDOW

        bvp_window = bvp[bvp_start:bvp_end]

        # Corresponding 700-Hz label window
        label_start = window_id * LABEL_WINDOW
        label_end = label_start + LABEL_WINDOW

        label_window = labels[label_start:label_end]

        # ----------------------------------------------------
        # Determine dominant WESAD condition
        # ----------------------------------------------------

        unique, counts = np.unique(label_window, return_counts=True)

        dominant_label = int(unique[np.argmax(counts)])

        # Ignore unwanted conditions
        if dominant_label in IGNORE_LABELS:
            continue

        # Convert to binary classification
        if dominant_label == STRESS_LABEL:
            binary_label = 1

        elif dominant_label in NON_STRESS_LABELS:
            binary_label = 0

        else:
            continue

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        X_all.append(bvp_window)
        y_all.append(binary_label)

        metadata.append({
            "subject": subject,
            "window_id": window_id,
            "start_sample": bvp_start,
            "end_sample": bvp_end,
            "start_time_seconds": bvp_start / BVP_FS,
            "end_time_seconds": bvp_end / BVP_FS,
            "wesad_label": dominant_label,
            "stress_label": binary_label
        })

        subject_count += 1

    print(f"Accepted windows: {subject_count}")


# ============================================================
# Convert to arrays
# ============================================================

X = np.asarray(X_all, dtype=np.float32)
y = np.asarray(y_all, dtype=np.int64)

metadata_df = pd.DataFrame(metadata)

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("metadata shape:", metadata_df.shape)

print("\nClass distribution:")

class_counts = metadata_df["stress_label"].value_counts().sort_index()

for label, count in class_counts.items():

    name = "NON-STRESS" if label == 0 else "STRESS"

    print(f"{label} ({name}): {count}")


print("\nSubjects:")
print(metadata_df["subject"].value_counts().sort_index())


# ============================================================
# Save
# ============================================================

np.save(OUTPUT / "X.npy", X)
np.save(OUTPUT / "y.npy", y)

metadata_df.to_csv(
    OUTPUT / "metadata.csv",
    index=False
)

print("\nSaved:")
print(OUTPUT / "X.npy")
print(OUTPUT / "y.npy")
print(OUTPUT / "metadata.csv")

print("\n✅ WESAD preprocessing completed successfully!")