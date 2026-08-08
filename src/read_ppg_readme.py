from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PPG-DaLiA"
    / "data"
    / "PPG_FieldStudy"
    / "PPG_FieldStudy_readme.pdf"
)

print("Reading:")
print(PDF_FILE)
print("=" * 70)

reader = PdfReader(str(PDF_FILE))

print("Number of pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages, start=1):
    print(f"\n{'=' * 70}")
    print(f"PAGE {page_number}")
    print("=" * 70)

    text = page.extract_text()

    if text:
        print(text)
    else:
        print("[No text extracted from this page]")