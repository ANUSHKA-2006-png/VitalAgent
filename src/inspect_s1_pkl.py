from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent

S1_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PPG-DaLiA"
    / "data"
    / "PPG_FieldStudy"
    / "S1"
)

PKL_FILE = S1_PATH / "S1.pkl"

print("Loading:", PKL_FILE)
print("=" * 60)

with open(PKL_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Type of loaded data:")
print(type(data))

print("\nTop-level information:")

if isinstance(data, dict):
    print("Dictionary keys:")
    for key in data.keys():
        print(" -", key)

else:
    print("Object attributes:")
    print(dir(data))