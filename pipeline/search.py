import os
import json
import sys
import subprocess
import numpy as np

MODEL = os.path.expanduser("~/adtc/adtc-2026-submission-template/model/all-MiniLM-L6-v2-ggml-model-f16.gguf")
STORE_FILE = os.path.expanduser("~/adtc/adtc-2026-submission-template/vector_store.json")


def embed(text):
    # Embed one piece of text with the SAME settings used to build the store
    result = subprocess.run(
        [
            "llama-embedding",
            "-m", MODEL,
            "-p", text,
            "--pooling", "mean",
            "--embd-normalize", "2",
            "--embd-output-format", "json",
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    return np.array(data["data"][0]["embedding"])


# Load the store
with open(STORE_FILE, encoding="utf-8") as f:
    store = json.load(f)

vectors = np.array([item["embedding"] for item in store])

# Get the question from the command line
question = sys.argv[1]
q_vec = embed(question)

# Similarity = dot product (vectors are normalized, so this is cosine similarity)
scores = vectors @ q_vec

# Find the top 3
top_idx = np.argsort(scores)[::-1][:3]

print(f"QUESTION: {question}")
print()
for rank, i in enumerate(top_idx, 1):
    print(f"--- MATCH {rank} (score {scores[i]:.3f}, from {store[i]['source']}) ---")
    print(store[i]["text"][:400])
    print()
