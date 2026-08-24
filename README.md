# Offline Cassava Advisor

An offline agricultural advisory for Nigerian cassava farmers and extension
workers. Runs entirely on an 8 GB laptop with no internet connection, no API
keys, and no cost per question.

The model answers questions about cassava pests, diseases, planting, weed
management, yield and variety selection. Every answer is grounded in passages
retrieved from vetted guides published by IITA, FAO, NAERLS and the Africa Soil
Health Consortium.

**ADTC 2026 submission** · Domain: Agriculture · Team ID: `offline-cassava-advisor`

---

## How this system works

The shipped GGUF is Llama-3.2-1B-Instruct with a condensed cassava knowledge digest
embedded in its chat template, so it answers cassava questions even when loaded bare
in LM Studio or Ollama with no system prompt supplied.
What makes it an advisor is the retrieval layer around it:

1. The farmer's question is expanded through a vocabulary map that translates
   everyday and Igbo farming terms into the technical language the guides use
   (for example "curling near the top" becomes "bunchy top, clumping, shoot tip").
2. The expanded question is embedded offline using llama.cpp and matched against
   677 passages from the source guides.
3. If nothing scores above the relevance threshold, the system declines to answer
   rather than guessing.
4. The top passages are passed to the model, which answers using only those facts.
5. Any chemical or treatment mentioned in the answer is flagged for the farmer to
   verify with an extension officer.

The cross-disciplinary integration is this offline RAG pipeline over agricultural
records. It is load-bearing: without it the model gives generic answers.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/ChukwumaUk/adtc-2026-submission-template.git
cd adtc-2026-submission-template
./download_model.sh          # fetches both model files (~815 MB total)
pip install numpy rank_bm25
```

You also need llama.cpp on your PATH:

```bash
brew install llama.cpp       # macOS
# or build from https://github.com/ggerganov/llama.cpp
```

Internet is needed for this step only. Everything after this runs offline.

### 2. Ask a question from the terminal

```bash
python advise.py "My cassava leaves are turning yellow and curling near the top of the plant. What could be causing this and what should I do?"
```

This prints the retrieved sources with their relevance scores, then the grounded
answer, then any safety warning. This is the simplest way to evaluate the system.

### 3. Or use the web interface

Two terminals are needed.

```bash
# Terminal 1: the model server
llama-server -m model/cassava-advisor-1B-Q4_K_M.gguf --port 8080 -c 4096

# Terminal 2: the web app
pip install fastapi uvicorn httpx
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. The interface streams the answer as it is generated,
shows which guides were consulted, and displays safety warnings prominently.

---

## Example questions

```bash
python advise.py "Weeds are taking over my cassava farm. How can I control them?"
python advise.py "Which cassava variety gives the highest yield?"
python advise.py "How can I increase the starch content of my cassava?"
python advise.py "akwukwo akpu m na-acha edo edo ma na-ekpoko onu"
python advise.py "How often should I vaccinate my goats?"
```

The Igbo question is a farmer describing yellowing, bunching cassava leaves.
The goat question is out of scope and is declined rather than answered.

---

## What is in this repository

| Path | Purpose |
|---|---|
| `advise.py` | The advisor. Retrieval, grounding, safety checks, and the CLI. |
| `app.py` | FastAPI server for the web interface. |
| `symptom_map.py` | Farmer vocabulary to technical vocabulary bridge, English and Igbo. |
| `vector_store.json` | 677 pre-computed passage embeddings. |
| `corpus/` | Cleaned source text, so the vector store is reproducible. |
| `pipeline/` | Scripts that built the corpus and the vector store. |
| `static/` | Web interface. No external assets. |
| `download_model.sh` | Fetches both model files. |
| `REPORT.md` | Technical report. |

---

## Rebuilding the vector store

Not required to run the system, but the pipeline is included for verification:

```bash
python pipeline/chunk_corpus.py
python pipeline/build_store.py
```

---

## Sources

- IITA, Pest Control in Cassava Farms (IPM Field Guide)
- IITA, Disease Control in Cassava Farms (IPM Field Guide)
- IITA BASICS-II, Profiles of Best Performing Improved Cassava Varieties
- NAERLS, Cassava Production, Processing and Utilization
- Africa Soil Health Consortium, Cassava System Cropping Guide
- FAO, Save and Grow: Cassava
