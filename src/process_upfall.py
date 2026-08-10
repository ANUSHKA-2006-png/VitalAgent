from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_UPFALL = PROJECT_ROOT / "data" / "raw" / "Fall_UP_Dataset" / "UP_Fall_Detection_Dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "fall"
SPLIT_DIR = OUTPUT_DIR / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS = 50
WINDOW_SECONDS = 2
WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS # 100 samples
MOMENT_INPUT_SIZE = 512

FALL_ACTIVITIES = {"A06", "A07", "A08", "A09", "A10", "A11"}
ADL_ACTIVITIES = {"A01", "A02", "A03", "A04", "A05"}

def main():
    print("=" * 70)
    print("VITALAGENT — UP-FALL DETECTION PREPROCESSING")
    print("=" * 70)

    all_windows = []
    all_labels = []
    metadata = []

    subject_folders = sorted([d for d in RAW_UPFALL.iterdir() if d.is_dir() and d.name.startswith("Subject")])
    print(f"Found UP-Fall subjects: {[d.name for d in subject_folders]}")

    for subject_folder in subject_folders:
        subject_id = subject_folder.name
        csv_files = sorted(subject_folder.glob("*/*.csv"))
        print(f"\nProcessing {subject_id} ({len(csv_files)} trial files)...")

        for csv_file in csv_files:
            activity = csv_file.parent.name
            if activity in FALL_ACTIVITIES:
                label = 1 # Fall
            elif activity in ADL_ACTIVITIES:
                label = 0 # Daily activity
            else:
                continue

            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"[WARNING] Error reading {csv_file}: {e}")
                continue

            df.columns = [c.strip() for c in df.columns]

            # Try Wrist acc first, fallback to Belt acc
            if "WRST_ACC_X" in df.columns and "WRST_ACC_Y" in df.columns and "WRST_ACC_Z" in df.columns:
                ax = df["WRST_ACC_X"].to_numpy(dtype=np.float32)
                ay = df["WRST_ACC_Y"].to_numpy(dtype=np.float32)
                az = df["WRST_ACC_Z"].to_numpy(dtype=np.float32)
            elif "BELT_ACC_X" in df.columns and "BELT_ACC_Y" in df.columns and "BELT_ACC_Z" in df.columns:
                ax = df["BELT_ACC_X"].to_numpy(dtype=np.float32)
                ay = df["BELT_ACC_Y"].to_numpy(dtype=np.float32)
                az = df["BELT_ACC_Z"].to_numpy(dtype=np.float32)
            else:
                continue

            if len(ax) < 10:
                continue

            # Compute resultant magnitude
            mag = np.sqrt(ax**2 + ay**2 + az**2)

            # Resample trial to 50 Hz equivalent
            num_resampled = max(WINDOW_SIZE, int(round(len(mag) * (TARGET_FS / 18.0))))
            mag_50hz = resample(mag, num_resampled).astype(np.float32)

            # Slice into 100-sample windows (2 sec) with 50% overlap (50 samples shift)
            shift = WINDOW_SIZE // 2
            start = 0
            w_idx = 0

            while start + WINDOW_SIZE <= len(mag_50hz):
                end = start + WINDOW_SIZE
                window = mag_50hz[start:end].copy()

                # Z-score normalize
                std = float(np.std(window))
                if std == 0:
                    start += shift
                    w_idx += 1
                    continue
                norm_window = (window - float(np.mean(window))) / std

                # Pad to 512 samples for MOMENT
                padded_window = np.pad(norm_window, (0, MOMENT_INPUT_SIZE - len(norm_window)), mode="constant")

                all_windows.append(padded_window.astype(np.float32))
                all_labels.append(label)
                metadata.append({
                    "subject": subject_id,
                    "activity": activity,
                    "trial": csv_file.stem,
                    "window_id": w_idx,
                    "fall_label": label
                })

                start += shift
                w_idx += 1

    X = np.asarray(all_windows, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int64)
    meta_df = pd.DataFrame(metadata)

    print("\n" + "=" * 70)
    print("FINAL FALL DATASET")
    print("=" * 70)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("metadata shape:", meta_df.shape)
    print("Class distribution (0=ADL, 1=Fall):")
    print(meta_df["fall_label"].value_counts())

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    meta_df.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

    # Subject-wise splits: S01 & S02 for train, S03 for val, S04 for test
    train_mask = meta_df["subject"].isin(["Subject_01", "Subject_02"]).to_numpy()
    val_mask = meta_df["subject"].isin(["Subject_03"]).to_numpy()
    test_mask = meta_df["subject"].isin(["Subject_04"]).to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    meta_train = meta_df[train_mask]
    meta_val = meta_df[val_mask]
    meta_test = meta_df[test_mask]

    np.save(SPLIT_DIR / "X_train.npy", X_train)
    np.save(SPLIT_DIR / "y_train.npy", y_train)
    meta_train.to_csv(SPLIT_DIR / "metadata_train.csv", index=False)

    np.save(SPLIT_DIR / "X_val.npy", X_val)
    np.save(SPLIT_DIR / "y_val.npy", y_val)
    meta_val.to_csv(SPLIT_DIR / "metadata_val.csv", index=False)

    np.save(SPLIT_DIR / "X_test.npy", X_test)
    np.save(SPLIT_DIR / "y_test.npy", y_test)
    meta_test.to_csv(SPLIT_DIR / "metadata_test.csv", index=False)

    print("\n[OK] Saved Fall dataset and splits:")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    print("\n[OK] UP-Fall preprocessing completed successfully!")

if __name__ == "__main__":
    main()
