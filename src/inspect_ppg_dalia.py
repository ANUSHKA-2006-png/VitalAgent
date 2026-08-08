from pathlib import Path

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

print("PPG-DaLiA — Subject S1")
print("=" * 60)

for item in sorted(S1_PATH.iterdir()):
    if item.is_dir():
        print(f"📁 {item.name}/")
    else:
        print(f"📄 {item.name}")