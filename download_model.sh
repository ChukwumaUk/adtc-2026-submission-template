#!/usr/bin/env bash
# Download the model weights for the Offline Cassava Advisor.
#
# Two models are required:
#   1. Llama-3.2-1B-Instruct-Q4_K_M.gguf  — the generator (declared in metadata.json)
#   2. all-MiniLM-L6-v2-ggml-model-f16.gguf — the embedder used for offline retrieval
#
# Rules:
#   - Idempotent (safe to run multiple times).
#   - No credentials required (public URLs only).
#   - Output paths match `_runtime.model_path` in metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
mkdir -p "$MODEL_DIR"

fetch() {
  local url="$1"
  local dest="$2"
  local label="$3"

  if [[ -f "$dest" ]]; then
    echo "$label already present at $dest — skipping"
    return 0
  fi

  echo "downloading $label → $dest"
  if command -v curl > /dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$dest.partial" "$url"
  elif command -v wget > /dev/null 2>&1; then
    wget --show-progress -O "$dest.partial" "$url"
  else
    echo "error: neither curl nor wget found" >&2
    exit 1
  fi
  mv "$dest.partial" "$dest"
  echo "done: $dest"
}

# 1. Generator model (~770 MB)
fetch \
  "https://huggingface.co/ChukwumaUk/cassava-advisor-1B-Q4_K_M/resolve/main/cassava-advisor-1B-Q4_K_M.gguf" \
  "$MODEL_DIR/cassava-advisor-1B-Q4_K_M.gguf" \
  "Cassava Advisor 1B Q4_K_M (generator)"

# 2. Embedding model (~44 MB)
fetch \
  "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-ggml-model-f16.gguf" \
  "$MODEL_DIR/all-MiniLM-L6-v2-ggml-model-f16.gguf" \
  "all-MiniLM-L6-v2-f16 (embedder)"

echo
echo "All models ready in $MODEL_DIR"
