import os
import json
import subprocess

CHUNKS_FILE = os.path.expanduser("~/adtc/workshop/corpus-clean/chunks.json")
MODEL = os.path.expanduser("~/adtc/adtc-2026-submission-template/model/all-MiniLM-L6-v2-ggml-model-f16.gguf")
STORE_FILE = os.path.expanduser("~/adtc/adtc-2026-submission-template/vector_store.json")

TEXTS_TMP = "chunk_texts.tmp"

# 1. Load the chunks
with open(CHUNKS_FILE, encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks.")

# 2. Write all chunk texts to a temp file, one per line.
#    Newlines inside a chunk would break the "one per line" rule,
#    so we flatten any internal newlines to spaces first.
with open(TEXTS_TMP, "w", encoding="utf-8") as f:
    for c in chunks:
        flat = " ".join(c["text"].split())
        f.write(flat + "\n")

# 3. Run llama-embedding once over the whole file
print("Embedding all chunks (this runs the model once)...")
result = subprocess.run(
    [
        "llama-embedding",
        "-m", MODEL,
        "-f", TEXTS_TMP,
        "--pooling", "mean",
        "--embd-normalize", "2",
        "--embd-output-format", "json",
    ],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("ERROR from llama-embedding:")
    print(result.stderr[-2000:])
    raise SystemExit(1)

# 4. Parse the JSON output (OpenAI style: {"data": [{"embedding": [...]}, ...]})
data = json.loads(result.stdout)
vectors = [item["embedding"] for item in data["data"]]
print(f"Got {len(vectors)} embeddings.")

if len(vectors) != len(chunks):
    print(f"WARNING: {len(chunks)} chunks but {len(vectors)} embeddings. Mismatch!")

# 5. Pair each chunk with its embedding and save
store = []
for chunk, vec in zip(chunks, vectors):
    store.append({
        "id": chunk["id"],
        "source": chunk["source"],
        "text": chunk["text"],
        "embedding": vec,
    })

with open(STORE_FILE, "w", encoding="utf-8") as f:
    json.dump(store, f)

os.remove(TEXTS_TMP)

print(f"Vector store written to: {STORE_FILE}")
print(f"Dimensions per embedding: {len(vectors[0]) if vectors else 'n/a'}")
