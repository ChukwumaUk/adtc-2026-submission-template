from pypdf import PdfReader
import os
import re

CORPUS_DIR = os.path.expanduser("~/adtc/workshop/corpus-raw")
OUTPUT_DIR = os.path.expanduser("~/adtc/workshop/corpus-clean")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Page ranges of real content, as human page numbers (1-based, inclusive)
PAGE_RANGES = {
    "iita-disease-control-cassava":        (4, 14),
    "iita-pest-control-cassava":           (4, 19),
    "naerls-cassava-production-processing": (4, 20),
    "ashc-cassava-cropping-guide":         (6, 60),
    "fao-save-and-grow-cassava":           (15, 120),
}


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


for name, (start, end) in PAGE_RANGES.items():
    path = os.path.join(CORPUS_DIR, name + ".pdf")
    reader = PdfReader(path)
    total = len(reader.pages)

    # Convert human page numbers to zero-based indices
    first = start - 1
    last = end  # slice end is exclusive, so this includes page `end`

    pages = []
    for page in reader.pages[first:last]:
        pages.append(page.extract_text() or "")

    full = "\n\n".join(pages)
    cleaned = clean_text(full)

    out_path = os.path.join(OUTPUT_DIR, name + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    kept = last - first
    print(f"{name}: kept pages {start}-{end} of {total} ({kept} pages) -> {len(cleaned)} chars")

print()
print("Trimmed and cleaned text written to:", OUTPUT_DIR)
