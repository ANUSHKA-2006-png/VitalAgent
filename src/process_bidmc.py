from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BIDMC = PROJECT_ROOT / "data" / "raw" / "BIDMC" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "spo2"
SPLIT_DIR = OUTPUT_DIR / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_FS = 125
TARGET_FS = 64
WINDOW_SECONDS = 8
SHIFT_SECONDS = 2

WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS
SHIFT_SIZE = TARGET_FS * SHIFT_SECONDS

# Butterworth bandpass filter: 0.5 - 4.0 Hz
def apply_bandpass_filter(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 4.0 / nyquist
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)

TRAIN_SUBJECTS = [f"bidmc_{i:02d}" for i in range(1, 38)]
VAL_SUBJECTS = [f"bidmc_{i:02d}" for i in range(38, 46)]
TEST_SUBJECTS = [f"bidmc_{i:02d}" for i in range(46, 54)]

def main():
    print("=" * 70)
    print("VITALAGENT — BIDMC SpO2 PREPROCESSING")
    print("=" * 70)

    all_windows = []
    all_labels = []
    metadata = []

    subject_signal_files = sorted(RAW_BIDMC.glob("*_Signals.csv"))
    print(f"Found BIDMC subject signal files: {len(subject_signal_files)}")

    for signal_file in subject_signal_files:
        subject_id = signal_file.name.replace("_Signals.csv", "")
        numerics_file = RAW_BIDMC / f"{subject_id}_Numerics.csv"

        if not numerics_file.exists():
            print(f"⚠️ Missing numerics file for {subject_id}")
            continue

        signals_df = pd.read_csv(signal_file)
        numerics_df = pd.read_csv(numerics_file)

        # Strip whitespace from column names
        signals_df.columns = [c.strip() for c in signals_df.columns]
        numerics_df.columns = [c.strip() for c in numerics_df.columns]

        if "PLETH" not in signals_df.columns or "SpO2" not in numerics_df.columns:
            print(f"⚠️ Required columns missing in {subject_id}")
            continue

        pleth = signals_df["PLETH"].to_numpy(dtype=np.float32)
        spo2_series = numerics_df["SpO2"].to_numpy(dtype=np.float32)
        numerics_time = numerics_df["Time [s]"].to_numpy(dtype=np.float32)

        # Resample PLETH from 125 Hz to 64 Hz
        pleth_64hz = resample_poly(pleth, TARGET_FS, ORIGINAL_FS).astype(np.float32)
        pleth_filtered = apply_bandpass_filter(pleth_64hz, TARGET_FS)

        total_samples = len(pleth_filtered)
        start = 0
        window_id = 0
        accepted = 0

        while start + WINDOW_SIZE <= total_samples:
            end = start + WINDOW_SIZE
            window = pleth_filtered[start:end].copy()

            start_sec = start / TARGET_FS
            end_sec = end / TARGET_FS

            # Match corresponding SpO2 values in the numerics table
            mask = (numerics_time >= start_sec) & (numerics_time <= end_sec)
            matching_spo2 = spo2_series[mask]

            if len(matching_spo2) == 0 or np.isnan(matching_spo2).all():
                start += SHIFT_SIZE
                window_id += 1
                continue

            mean_spo2 = float(np.nanmean(matching_spo2))

            # Filter out invalid readings (< 50% or > 100%)
            if mean_spo2 < 50.0 or mean_spo2 > 100.0:
                start += SHIFT_SIZE
                window_id += 1
                continue

            # Z-score normalize window
            std = float(np.std(window))
            if std == 0:
                start += SHIFT_SIZE
                window_id += 1
                continue

            norm_window = (window - float(np.mean(window))) / std

            # Ensure exact length of 512 samples
            if len(norm_window) < 512:
                norm_window = np.pad(norm_window, (0, 512 - len(norm_window)), mode="constant")
            elif len(norm_window) > 512:
                norm_window = norm_window[:512]

            all_windows.append(norm_window.astype(np.float32))
            all_labels.append(mean_spo2)
            metadata.append({
                "subject": subject_id,
                "window_id": window_id,
                "start_time_seconds": start_sec,
                "end_time_seconds": end_sec,
                "spo2_pct": mean_spo2
            })

            start += SHIFT_SIZE
            window_id += 1
            accepted += 1

        print(f"Processed {subject_id}: {accepted} windows accepted.")

    X = np.asarray(all_windows, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.float32)
    meta_df = pd.DataFrame(metadata)

    print("\n" + "=" * 70)
    print("FINAL SpO2 DATASET")
    print("=" * 70)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("metadata shape:", meta_df.shape)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    meta_df.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

    # Subject-wise splits
    train_mask = meta_df["subject"].isin(TRAIN_SUBJECTS).to_numpy()
    val_mask = meta_df["subject"].isin(VAL_SUBJECTS).to_numpy()
    test_mask = meta_df["subject"].isin(TEST_SUBJECTS).to_numpy()

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

    print("\n[OK] Saved SpO2 dataset and splits:")
    print(f"Train: X={X_train.shape}, y={y_train.shape} (Subjects: {len(TRAIN_SUBJECTS)})")
    print(f"Val:   X={X_val.shape}, y={y_val.shape} (Subjects: {len(VAL_SUBJECTS)})")
    print(f"Test:  X={X_test.shape}, y={y_test.shape} (Subjects: {len(TEST_SUBJECTS)})")
    print("\n[OK] BIDMC SpO2 preprocessing completed successfully!")

if __name__ == "__main__":
    main()
