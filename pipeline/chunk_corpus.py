import os
import re
import glob
import json

CLEAN_DIR = os.path.expanduser("~/adtc/workshop/corpus-clean")
OUT_FILE = os.path.expanduser("~/adtc/workshop/corpus-clean/chunks.json")

CHUNK_SIZE = 800
OVERLAP = 150


def split_into_chunks(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size

        # Try not to cut mid-sentence: look for the last period in the tail
        if end < len(text):
            window = text[start:end]
            last_period = window.rfind(". ")
            if last_period > size * 0.5:   # only if it's not too early
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def is_junk(chunk):
    # Too short to carry meaning
    if len(chunk) < 150:
        return True

    # Mostly digits and punctuation (tables, page furniture)
    letters = sum(c.isalpha() for c in chunk)
    if letters / max(len(chunk), 1) < 0.5:
        return True

    return False


all_chunks = []

for path in sorted(glob.glob(os.path.join(CLEAN_DIR, "*.txt"))):
    source = os.path.basename(path).replace(".txt", "")
    with open(path, encoding="utf-8") as f:
        text = f.read()

    raw_chunks = split_into_chunks(text, CHUNK_SIZE, OVERLAP)
    kept = [c for c in raw_chunks if not is_junk(c)]
    dropped = len(raw_chunks) - len(kept)

    for i, chunk in enumerate(kept):
        all_chunks.append({
            "id": f"{source}--{i}",
            "source": source,
            "text": chunk,
        })

    print(f"{source}: {len(raw_chunks)} chunks, dropped {dropped} junk, kept {len(kept)}")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print()
print(f"TOTAL CHUNKS: {len(all_chunks)}")
print("Written to:", OUT_FILE)
