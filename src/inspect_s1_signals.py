from pathlib import Path
import pickle

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

with open(PKL_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

signals = data["signal"]

print("=" * 70)
print("PPG-DaLiA — S1 SIGNAL INSPECTION")
print("=" * 70)

for body_location, signal_data in signals.items():

    print(f"\n[{body_location.upper()}]")
    print("-" * 50)

    print("Type:", type(signal_data))

    if isinstance(signal_data, dict):

        print("Keys:")

        for key, value in signal_data.items():

            print(f"\n  {key}")
            print(f"    Type: {type(value)}")

            if hasattr(value, "shape"):
                print(f"    Shape: {value.shape}")

            if hasattr(value, "dtype"):
                print(f"    Data type: {value.dtype}")

    else:

        if hasattr(signal_data, "shape"):
            print("Shape:", signal_data.shape)

        if hasattr(signal_data, "dtype"):
            print("Data type:", signal_data.dtype)