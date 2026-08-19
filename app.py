"""Local web interface for the offline cassava advisor.

Serves a single-page UI and streams answers over Server-Sent Events.

Retrieval, the prompt, the relevance gate and the chemical check are imported
from advise.py unchanged. The only difference from the CLI is generation:
this talks to a running llama-server instead of spawning llama-cli per
request, so tokens can be streamed to the browser as they are produced.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import advise

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "http://localhost:8080")

# Sampling is pinned so the web answer matches the CLI answer exactly.
TEMPERATURE = 0
SEED = 42
N_PREDICT = 600

# llama-cli starts a fresh process per question, so it always evaluates the
# prompt cold. llama-server reuses its KV cache across requests, and the
# reused path drifts: on the yellow-leaf question it added fungicide and neem
# oil recommendations the cold path does not make. Reprocessing the prompt
# costs about 20 seconds and buys answers identical to the safety-tested CLI.
CACHE_PROMPT = False

MAX_QUESTION_CHARS = 2000

# llama-cli runs an instruct model in conversation mode, so the prompt it
# sends is wrapped in the model's chat template. llama-server's /completion
# does no wrapping. Without applying the same template the model answers a
# noticeably different and less grounded question, so we ask the server for
# its own template rather than guessing.
FALLBACK_TEMPLATE = (
    "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

KNOWLEDGE = {}


@asynccontextmanager
async def lifespan(app):
    store, vectors, bm25 = advise.load_store()
    KNOWLEDGE.update(store=store, vectors=vectors, bm25=bm25)
    print(f"Loaded {len(store)} passages. Generating via {LLAMA_SERVER}")
    yield


app = FastAPI(title="Cassava Advisor", lifespan=lifespan)


class Question(BaseModel):
    question: str


def sse(event, data):
    """One Server-Sent Event. Data is always a JSON object."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def apply_chat_template(client, prompt):
    try:
        response = await client.post(
            f"{LLAMA_SERVER}/apply-template",
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()["prompt"]
    except Exception:
        # Older llama-server builds have no /apply-template.
        return FALLBACK_TEMPLATE.replace("{prompt}", prompt)


async def answer_stream(request, question):
    """Yield the full lifecycle of one question as typed SSE events.

    Once this generator starts, the response is already 200 OK, so every
    failure below is reported as an `error` event rather than a status code.
    """
    started = time.monotonic()

    question = question.strip()
    if not question:
        yield sse("error", {"message": "Type a question first."})
        return
    if len(question) > MAX_QUESTION_CHARS:
        yield sse("error", {
            "message": f"That question is too long. Keep it under "
                       f"{MAX_QUESTION_CHARS} characters."
        })
        return

    # --- Retrieval -------------------------------------------------------
    yield sse("stage", {"stage": "retrieving"})
    try:
        ranked = await asyncio.to_thread(
            advise.retrieve,
            question,
            KNOWLEDGE["store"],
            KNOWLEDGE["vectors"],
            KNOWLEDGE["bm25"],
        )
    except FileNotFoundError:
        yield sse("error", {
            "message": "Could not run llama-embedding. Check that llama.cpp "
                       "is installed and on your PATH."
        })
        return
    except Exception as exc:
        yield sse("error", {"message": f"Could not search the guides: {exc}"})
        return

    # --- Relevance gate --------------------------------------------------
    # Same gate as the CLI: the best raw semantic score, not the fused score.
    best_sem = max(sem for _, sem, _ in ranked)
    if best_sem < advise.RELEVANCE_THRESHOLD:
        yield sse("token", {"text": advise.OFF_TOPIC_MESSAGE})
        yield sse("done", {
            "grounded": False,
            "sources": 0,
            "elapsed": round(time.monotonic() - started, 1),
        })
        return

    # Document name and relevance only. Chunk text is deliberately not sent.
    yield sse("sources", {"sources": [
        {"document": chunk["source"], "score": round(sem, 3)}
        for chunk, sem, _ in ranked
    ]})

    # --- Generation ------------------------------------------------------
    yield sse("stage", {"stage": "generating"})

    pieces = []
    leading = True
    timeout = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            payload = {
                "prompt": await apply_chat_template(
                    client, advise.build_prompt(question, ranked)
                ),
                "temperature": TEMPERATURE,
                "seed": SEED,
                "n_predict": N_PREDICT,
                "stream": True,
                "cache_prompt": CACHE_PROMPT,
            }
            async with client.stream(
                "POST", f"{LLAMA_SERVER}/completion", json=payload
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    yield sse("error", {
                        "message": f"llama-server replied {response.status_code}. "
                                   f"Check the terminal running it."
                    })
                    return

                async for line in response.aiter_lines():
                    if await request.is_disconnected():
                        return
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    text = chunk.get("content", "")
                    if text:
                        if leading:
                            # The CLI strips the answer; match that.
                            text = text.lstrip()
                            if not text:
                                continue
                            leading = False
                        pieces.append(text)
                        yield sse("token", {"text": text})
                    if chunk.get("stop"):
                        break

    except httpx.ConnectError:
        yield sse("error", {
            "message": f"Cannot reach llama-server at {LLAMA_SERVER}. "
                       f"Start it, then ask again."
        })
        return
    except httpx.ReadTimeout:
        yield sse("error", {
            "message": "The model stopped responding. Ask again."
        })
        return
    except Exception as exc:
        yield sse("error", {"message": f"Generation failed: {exc}"})
        return

    answer = "".join(pieces).strip()

    # --- Chemical safety check, identical to the CLI ---------------------
    spans, invented = advise.find_invented_spans(answer, ranked)
    if invented:
        yield sse("redaction", {
            # The answer is sent back with the offsets that index into it. The
            # browser marks up this copy rather than the one it assembled from
            # tokens, so a redaction can never miss because the two strings
            # drifted apart.
            "answer": answer,
            "spans": [[start, end] for start, end in spans],
            "terms": invented,
            "replacement": (
                "[A treatment recommendation was removed here because it "
                "does not appear in the source guides.]"
            ),
            "message": (
                "One or more treatment recommendations were removed because "
                "they are not in the source guides: " + ", ".join(invented) +
                ". This system only passes on advice that appears in its "
                "sources. Please see your local agricultural extension officer "
                "before applying anything to your crop."
            ),
        })

    yield sse("done", {
        "grounded": True,
        "sources": len(ranked),
        "elapsed": round(time.monotonic() - started, 1),
    })


@app.post("/ask")
async def ask(request: Request, body: Question):
    return StreamingResponse(
        answer_stream(request, body.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status")
async def status():
    """Lets the page say the model is missing before a farmer waits on it."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLAMA_SERVER}/health")
            ready = response.status_code == 200
    except Exception:
        ready = False
    return {
        "model_ready": ready,
        "llama_server": LLAMA_SERVER,
        "passages": len(KNOWLEDGE.get("store", [])),
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
