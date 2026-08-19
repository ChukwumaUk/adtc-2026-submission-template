import os
import re
import json
import sys
import subprocess
import re
import numpy as np
from rank_bm25 import BM25Okapi
from symptom_map import expand_query

HERE = os.path.dirname(os.path.abspath(__file__))

EMBED_MODEL = os.path.join(HERE, "model", "all-MiniLM-L6-v2-ggml-model-f16.gguf")
GEN_MODEL = os.path.join(HERE, "model", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
STORE_FILE = os.path.join(HERE, "vector_store.json")

TOP_K = 4
RELEVANCE_THRESHOLD = 0.45
RRF_K = 60
SEM_WEIGHT = 1.0   # meaning search dominates
KW_WEIGHT = 0.0   # BM25 disabled: expansion alone surfaces the right chunks
SEM_FLOOR = 0.40   # keyword hits below this semantic score are discarded
CANDIDATES = 20

CHEMICAL_WORDS = [
    "fungicide", "pesticide", "insectic", "herbicid", "spray",
    "copper", "chlorothalonil", "mancozeb", "carbendazim",
    "dimethoate", "malathion", "neem oil", "vaccine", "antibiotic",
]

OFF_TOPIC_MESSAGE = (
    "I focus on cassava farming, and I could not find reliable "
    "information in my guides to answer this question. Please consult "
    "your local agricultural extension officer for advice on this topic."
)


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def embed(text):
    result = subprocess.run(
        ["llama-embedding", "-m", EMBED_MODEL, "-p", text,
         "--pooling", "mean", "--embd-normalize", "2",
         "--embd-output-format", "json"],
        capture_output=True, text=True,
    )
    return np.array(json.loads(result.stdout)["data"][0]["embedding"])


def hybrid_retrieve(question, store, vectors, bm25):
    """Search by meaning AND by keyword, then fuse the two rankings."""
    expanded = expand_query(question)

    # Semantic search
    q_vec = embed(expanded)
    sem_scores = vectors @ q_vec
    sem_ranked = np.argsort(sem_scores)[::-1][:CANDIDATES]

    # Keyword search
    kw_scores = bm25.get_scores(tokenize(expanded))
    kw_ranked = np.argsort(kw_scores)[::-1][:CANDIDATES]
    # BM25 may rank a chunk highly on shared common words alone.
    # Only let it promote chunks that are at least plausibly on-topic.
    kw_ranked = [i for i in kw_ranked if sem_scores[i] >= SEM_FLOOR]

    # Reciprocal Rank Fusion
    fused = {}
    for rank, idx in enumerate(sem_ranked, 1):
        fused[idx] = fused.get(idx, 0) + SEM_WEIGHT / (RRF_K + rank)
    for rank, idx in enumerate(kw_ranked, 1):
        fused[idx] = fused.get(idx, 0) + KW_WEIGHT / (RRF_K + rank)

    final = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

    # Return chunks with BOTH their fused score and their raw semantic score.
    # The semantic score is what the relevance gate uses, since it is on a
    # meaningful 0-1 scale, while fused scores are not.
    return [(store[i], float(sem_scores[i]), float(fscore)) for i, fscore in final]


# Alias so callers can import the retrieval step under its generic name.
retrieve = hybrid_retrieve


SYMPTOM_SIGNALS = [
    "turning", "yellow", "yellowing", "dying", "died", "wilting", "curling",
    "curled", "spots", "spotted", "rotting", "rot", "holes", "drying",
    "distorted", "stunted", "bunchy", "clumping", "chlorotic", "necrosis",
    "what is wrong", "what's wrong", "problem with", "attacking", "damaged",
    "damage", "infested", "sick", "not growing", "falling off",
]


def classify_question(expanded_question):
    """Decide in code whether this is a diagnostic or advisory question.
    The model cannot classify reliably, so we do it here and send only the
    relevant instruction. Runs on the EXPANDED query so Igbo symptom terms,
    which expand into English, classify correctly too."""
    q = expanded_question.lower()
    return "diagnostic" if any(sig in q for sig in SYMPTOM_SIGNALS) else "advisory"


TREATMENT_SIGNALS = [
    "control", "manage", "treat", "apply", "spray", "remove", "uproot",
    "destroy", "rotate", "fallow", "resistant variety", "resistant varieties",
    "clean planting", "healthy cuttings", "biological control", "natural enemy",
    "predator", "hand weed", "weeding", "prevent", "avoid planting",
]


def facts_contain_treatment(ranked_chunks):
    """True if any retrieved passage actually says what to DO, not just what
    the problem looks like. Symptom-only passages invite the model to invent
    treatments, so we detect that case and instruct it explicitly."""
    joined = " ".join(c["text"].lower() for c, _, _ in ranked_chunks)
    return any(sig in joined for sig in TREATMENT_SIGNALS)


def build_prompt(question, ranked_chunks):
    facts = ""
    for n, (chunk, sem, fused) in enumerate(ranked_chunks, 1):
        facts += f"{n}. {chunk['text']}\n\n"

    # Classify in code, not in the prompt. A 1B model cannot reliably pick
    # between branching instructions, so we send only the relevant one.
    # Classify on the EXPANDED query so Igbo symptom terms classify correctly.
    kind = classify_question(expand_query(question))
    if kind == "diagnostic":
        how_to = ("- Name the single most likely cause of the problem first, taken from the FACTS, "
                  "and say briefly why. Then mention one other possible cause if the FACTS support one.")

    else:
        how_to = ("- The farmer is asking for guidance, not reporting a problem. Do NOT diagnose a cause "
                  "and do not begin with 'the most likely cause'. Give clear, practical steps or options "
                  "drawn from the FACTS.")

    return f"""You are an expert farming advisor for Nigerian cassava farmers. Below are facts retrieved from trusted agricultural guides, ordered from most to least relevant.

STRICT RULES:
- Use ONLY the information in the FACTS below. Do not add treatments, chemicals, products, or recommendations that are not explicitly stated in the FACTS.
- If the FACTS do not mention how to treat or solve the problem, say honestly that the guides do not specify a treatment, and advise the farmer to consult a local agricultural extension officer. Do NOT invent a solution.
- Do not give scientific or Latin names unless they appear in the FACTS.
- Do not recommend any chemical product or commercial treatment unless that exact product name is written in the FACTS.
- Never repeat the same recommendation more than once.

HOW TO ANSWER:
{how_to}
- Give the farmer every relevant option the FACTS support, not just one.
- Be clear, practical, and brief.

FACTS:
{facts}
QUESTION: {question}

YOUR ADVICE:"""


def generate(question, ranked_chunks):
    prompt = build_prompt(question, ranked_chunks)

    result = subprocess.run(
        ["llama-cli", "-m", GEN_MODEL, "-p", prompt,
         "-n", "600", "--no-warmup", "-st",
         "--temp", "0", "--seed", "42",
         "--no-display-prompt", "--no-show-timings", "--simple-io"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    raw = result.stdout
    # llama-cli echoes the prompt, so keep only what follows our answer marker
    if "YOUR ADVICE:" in raw:
        raw = raw.rsplit("YOUR ADVICE:", 1)[1]
    return raw.replace("Exiting...", "").strip()


SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def find_invented_spans(answer, ranked_chunks):
    """Return (spans, terms) locating invented treatments inside `answer`.

    Spans are (start, end) character offsets into the exact string passed in,
    so the caller can mark up that same string. Offsets rather than sentence
    text: the web path used to send the offending sentences and have the
    browser re-split the streamed answer and match them by string equality,
    which silently found nothing whenever the two splits disagreed - notably
    when the answer stopped without terminal punctuation, since only one side
    had been stripped. A failed match left the invented treatment on screen.
    Offsets remove the second derivation, so there is nothing left to disagree.

    The CLI keeps using strip_invented_treatments; this is the web path only.
    """
    facts_text = " ".join(c["text"].lower() for c, _, _ in ranked_chunks)

    bounds = []
    start = 0
    for match in SENTENCE_BREAK.finditer(answer):
        bounds.append((start, match.start()))
        start = match.end()
    bounds.append((start, len(answer)))

    spans, terms = [], []
    for begin, end in bounds:
        low = answer[begin:end].lower()
        hits = [w for w in CHEMICAL_WORDS if w in low and w not in facts_text]
        if not hits:
            continue
        # Tighten onto the visible text so the strike-through does not trail
        # across the blank line that follows a sentence.
        while end > begin and answer[end - 1].isspace():
            end -= 1
        while begin < end and answer[begin].isspace():
            begin += 1
        if begin < end:
            spans.append((begin, end))
            terms.extend(hits)

    seen = set()
    return spans, [t for t in terms if not (t in seen or seen.add(t))]


def strip_invented_treatments(answer, ranked_chunks):
    """Remove any line recommending a chemical that does not appear in the
    retrieved facts, and leave a visible marker in its place.

    A word list cannot predict when the model will invent a treatment, but it
    can identify one after the fact. This acts on what was actually produced
    rather than trying to forecast it. Returns (cleaned_answer, removed_terms).
    """
    facts_text = " ".join(c["text"].lower() for c, _, _ in ranked_chunks)
    removed = []
    kept_lines = []
    marker_added = False

    for line in answer.split("\n"):
        low = line.lower()
        hits = [w for w in CHEMICAL_WORDS if w in low and w not in facts_text]
        if hits:
            removed.extend(hits)
            if not marker_added:
                kept_lines.append(
                    "[A treatment recommendation was removed here because it "
                    "does not appear in the source guides.]"
                )
                marker_added = True
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()

    if removed:
        cleaned += (
            "\n\nThe guides consulted describe this problem but do not specify a "
            "treatment for it. Please see your local agricultural extension officer "
            "before applying anything to your crop."
        )

    seen = set()
    unique = [r for r in removed if not (r in seen or seen.add(r))]
    return cleaned, unique


def check_invented_chemicals(answer, ranked_chunks):
    """Flag ANY chemical or product mention. For smallholder farmers, a
    chemical recommendation always warrants caution, whether or not the
    word appears somewhere in the retrieved text."""
    answer_lower = answer.lower()
    return [w for w in CHEMICAL_WORDS if w in answer_lower]


def load_store():
    """Load everything once. Returns (store, vectors, bm25)."""
    with open(STORE_FILE, encoding="utf-8") as f:
        store = json.load(f)
    vectors = np.array([item["embedding"] for item in store])
    bm25 = BM25Okapi([tokenize(c["text"]) for c in store])
    return store, vectors, bm25


def main():
    store, vectors, bm25 = load_store()

    question = sys.argv[1]

    print("Retrieving knowledge...")
    ranked = hybrid_retrieve(question, store, vectors, bm25)
    print("Top facts (hybrid: meaning + keyword):")
    for n, (chunk, sem, fused) in enumerate(ranked, 1):
        print(f"  {n}. [sem {sem:.3f}] {chunk['source'][:22]:24s} {chunk['text'][:55]}...")
    print()
    print("=" * 70)
    print("ADVISOR'S ANSWER:")
    print("=" * 70)

    # Gate uses the best SEMANTIC score, which is on a meaningful scale
    best_sem = max(sem for _, sem, _ in ranked)
    if best_sem < RELEVANCE_THRESHOLD:
        print(OFF_TOPIC_MESSAGE)
    else:
        answer = generate(question, ranked)
        answer, removed = strip_invented_treatments(answer, ranked)
        print(answer)
        if removed:
            print()
            print("=" * 70)
            print("SAFETY NOTICE: One or more treatment recommendations were removed")
            print(f"because they are not in the source guides: {', '.join(removed)}")
            print("This system only passes on advice that appears in its sources.")


if __name__ == "__main__":
    main()
