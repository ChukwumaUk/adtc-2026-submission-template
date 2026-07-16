import os
import json
import sys
import subprocess
import numpy as np

# Find this script's own folder, so paths work no matter where the repo lives
HERE = os.path.dirname(os.path.abspath(__file__))

EMBED_MODEL = os.path.join(HERE, "model", "all-MiniLM-L6-v2-ggml-model-f16.gguf")
GEN_MODEL = os.path.join(HERE, "model", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
STORE_FILE = os.path.join(HERE, "vector_store.json")

TOP_K = 4
RELEVANCE_THRESHOLD = 0.45


def embed(text):
    result = subprocess.run(
        ["llama-embedding", "-m", EMBED_MODEL, "-p", text,
         "--pooling", "mean", "--embd-normalize", "2",
         "--embd-output-format", "json"],
        capture_output=True, text=True,
    )
    return np.array(json.loads(result.stdout)["data"][0]["embedding"])


def retrieve(question, store, vectors):
    q_vec = embed(question)
    scores = vectors @ q_vec
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [(store[i], float(scores[i])) for i in top_idx]


def generate(question, ranked_chunks):
    facts = ""
    for n, (chunk, score) in enumerate(ranked_chunks, 1):
        facts += f"{n}. {chunk['text']}\n\n"

    prompt = f"""You are an expert farming advisor for Nigerian cassava farmers. Below are facts retrieved from trusted agricultural guides, ordered from most to least relevant.

STRICT RULES:
- Use ONLY the information in the FACTS below. Do not add treatments, chemicals, products, or recommendations that are not explicitly stated in the FACTS.
- If the FACTS do not mention how to treat or solve the problem, say honestly that the guides do not specify a treatment, and advise the farmer to consult a local agricultural extension officer. Do NOT invent a solution.
- Never recommend a pesticide, fertilizer, or product unless it is named in the FACTS.

HOW TO ANSWER:
- If the question describes a problem, name the most likely cause first (from the FACTS), then briefly mention any alternative cause.
- If the question asks how to do something, give clear step-by-step guidance from the FACTS.
- Be clear, practical, and brief.

FACTS:
{facts}
QUESTION: {question}

ANSWER:"""

    result = subprocess.run(
        ["llama-cli", "-m", GEN_MODEL, "-p", prompt,
         "-n", "250", "--no-warmup", "-st",
         "--temp", "0.2", "--top-p", "0.9",
         "--no-display-prompt", "--no-show-timings"],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


with open(STORE_FILE, encoding="utf-8") as f:
    store = json.load(f)
vectors = np.array([item["embedding"] for item in store])

question = sys.argv[1]

print("Retrieving knowledge...")
ranked = retrieve(question, store, vectors)
print("Top facts by relevance:")
for n, (chunk, score) in enumerate(ranked, 1):
    print(f"  {n}. [{score:.3f}] {chunk['source']}: {chunk['text'][:60]}...")
print()
print("=" * 70)
print("ADVISOR'S ANSWER:")
print("=" * 70)

top_score = ranked[0][1]
if top_score < RELEVANCE_THRESHOLD:
    print(
        "I focus on cassava farming, and I could not find reliable "
        "information in my guides to answer this question. Please consult "
        "your local agricultural extension officer for advice on this topic."
    )
else:
    print(generate(question, ranked))
