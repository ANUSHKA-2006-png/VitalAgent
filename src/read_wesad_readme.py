from pathlib import Path
import fitz

README_PATH = Path("data/raw/WESAD/WESAD/wesad_readme.pdf")

print("=" * 70)
print("VITALAGENT — WESAD README")
print("=" * 70)

doc = fitz.open(README_PATH)

print(f"\nNumber of pages: {len(doc)}")

for i, page in enumerate(doc):
    print("\n" + "=" * 70)
    print(f"PAGE {i + 1}")
    print("=" * 70)

    text = page.get_text()
    print(text)