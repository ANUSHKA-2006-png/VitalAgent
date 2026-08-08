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

print("=" * 70)
print("PPG-DaLiA — S1.pkl CONTENT INSPECTION")
print("=" * 70)

for key, value in data.items():

    print(f"\n[{key}]")
    print("-" * 50)

    print("Type:", type(value))

    # Dictionary
    if isinstance(value, dict):
        print("Dictionary keys:")
        for subkey in value.keys():
            print("  -", subkey)

    # List / tuple
    elif isinstance(value, (list, tuple)):
        print("Length:", len(value))

        if len(value) > 0:
            print("First element type:", type(value[0]))

    # Anything with shape
    if hasattr(value, "shape"):
        print("Shape:", value.shape)

    # Anything with length
    elif hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            print("Length:", len(value))
        except TypeError:
            pass