from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"

datasets = [
    "PPG-DaLiA",
    "WESAD",
    "Fall_UP_Dataset",
    "BIDMC",
]

print("VITALAGENT DATASET CHECK")
print("=" * 40)

for dataset in datasets:
    path = RAW_DATA / dataset

    if path.exists():
        items = list(path.iterdir())

        print(f"\n✅ {dataset}")
        print(f"   Location: {path}")
        print(f"   Items: {len(items)}")

        for item in items[:5]:
            print(f"      - {item.name}")

    else:
        print(f"\n❌ {dataset} NOT FOUND")