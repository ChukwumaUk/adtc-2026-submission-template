# Technical Report — Offline Cassava Advisor

**Team ID:** offline-cassava-advisor
**Domain:** Agriculture
**Model:** cassava-advisor-1B-Q4_K_M (Llama-3.2-1B-Instruct, GGUF Q4_K_M, llama.cpp)
**Repository:** https://github.com/ChukwumaUk/adtc-2026-submission-template
**Model weights:** https://huggingface.co/ChukwumaUk/cassava-advisor-1B-Q4_K_M

---

## Problem

Nigeria is the largest cassava producer in the world, and cassava is grown by smallholder farmers across the country as both a food crop and a source of income. When something goes wrong in the field, the farmer needs to know what it is and what to do about it. The formal answer is to consult an agricultural extension officer, but there are far too few of them, they cannot reach every village, and a farmer standing in front of a sick plant needs an answer that day, not next month.

Before I chose what to build, I spoke to an agriculturist at the agricultural ministry in Abakaliki, Ebonyi State. She was the one who convinced me that agriculture was too broad a field to attack as a whole and that I needed to narrow to a single crop. She also told me something that shaped the problem definition itself. She belongs to an international agriculture WhatsApp group where farmers and experts talk, and she said farmers come into that group constantly with the same kind of question. Their crop is sick, they do not know what is wrong, and they are asking an expert to tell them what it is and what to do. The farmers asking were growing cocoa, maize, pepper, groundnut and rice, not cassava. That is what convinced me the problem was real. A farmer with a sick crop and no access to an extension officer will go to whatever channel they can reach, and a WhatsApp group full of strangers is what they have.

The obvious modern answer is to ask an AI assistant. That answer fails for the user I am building for. Cloud assistants need a stable connection, they need data the farmer pays for by the megabyte, and the better ones need a subscription in dollars. In a village in Ebonyi or Abia, all three of those are real barriers, and the first one is often absolute.

So the problem I set out to solve is not whether an AI can answer cassava questions. It is whether trustworthy cassava advice can be delivered with no internet, no data cost, and no subscription, on hardware that already exists in the places that need it.

I validated the direction with two further specialists. A production manager at Green Hills, a subsidiary of the Saro Africa Group that grows cassava for ethanol on a large farm in Edo State, told me their biggest problem is weed management, not disease, and that the two things they measure their crop on are yield and starch content. A PhD agricultural researcher with IITA experience, who visits villages to speak with farmers directly, reviewed the system and submitted twenty written recommendations. His most important observation was that most of the farmers he meets cannot read or write English.

Those conversations shaped the system directly. Weed management became a first-class concern. Variety selection became central, because it turns out to answer both the weed problem and the starch problem. And the language question moved from a nice-to-have to a design requirement.

---

## Constraints

**Hardware.** The target is an 8 GB laptop with integrated graphics and no discrete GPU. I did not simulate this constraint. My development machine is a 2015 MacBook Air with an Intel Core i5-5250U, a dual-core processor from eleven years ago, and 8 GB of RAM. Everything reported here was built and measured on hardware at or below the standard evaluation machine, so the numbers are pessimistic rather than flattering.

**Connectivity.** The system must answer with the network cable pulled out. This ruled out cloud inference, cloud embeddings, hosted vector databases, and any frontend that loads a font, a stylesheet or a script from a CDN.

**Data cost.** Even where there is a signal, data is expensive. A system that makes a network call per question is not free to run, and a farmer will notice. The offline requirement is about economics as much as availability.

**Vocabulary.** Agricultural guides are written by agronomists. Farmers do not speak that way. A guide describes leaves that clump together into bunchy tops while a farmer says the top of the plant is bunching, or says it in Igbo. This gap broke retrieval repeatedly during development and became one of the harder problems in the build.

**Model capability.** A model small enough to run comfortably on this hardware has real limits. A 1 billion parameter model does not reason reliably, does not follow complex instructions consistently, and will invent plausible-sounding detail when it has room to. I treated this as a design constraint to engineer around rather than a flaw to apologise for, and the safety architecture exists because of it.

---

## Design Decisions

Every significant choice was made by testing alternatives on the target hardware rather than by reading benchmarks written for machines I do not have.

**Base model.** I tested four generators: Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct and Llama-3.2-3B-Instruct. I started with Qwen because its documentation emphasises instruction following. I abandoned it when testing showed it recommending fungicide for cassava mosaic disease, which is a virus. No fungicide treats a virus, and telling a smallholder to spend money on one is worse than telling them nothing. Llama-3.2-1B did not make that error under the same conditions. SmolLM2 echoed the prompt back instead of answering when retested on the corrected pipeline. Llama-3.2-3B followed rules cleanly but inverted a comparative sentence from its own source, misread a how-to question as a diagnosis, and took roughly two minutes per answer, which no farmer will wait for.

**Quantization.** I benchmarked Q3_K_L, Q4_K_M and Q8_0 on the development machine. Q4_K_M measured fastest at 18.4 tokens per second. Q3_K_L, despite being the smallest file, measured 11 tokens per second, because 4-bit K-quant kernels are more optimised than 3-bit ones on CPU. Q8_0 measured 11.15. The smallest file was not the fastest, which is the kind of thing you only learn by measuring.

**Retrieval.** Semantic search over 688 passages using all-MiniLM-L6-v2 in GGUF, embedded through llama.cpp so the system has no Python ML dependency at query time. I then added BM25 keyword search fused by Reciprocal Rank Fusion, and later removed it. Controlled testing showed that query expansion alone surfaced the passages BM25 was meant to find, while BM25 was displacing correct passages on yield questions. The simpler system retrieved better.

**Generation settings.** Temperature 0 with a fixed seed, so the same question always produces the same answer. I briefly added a repetition penalty to stop a looping failure and removed it when bisection showed it was penalising the model for reusing vocabulary from the retrieved passages, which is precisely what grounding requires. Loop protection comes from the token limit instead.

**Embedded knowledge digest.** Late in development the organisers clarified that judges test the model by direct inference, without the surrounding application. A bare Llama-3.2-1B answering a cassava question calls cassava mosaic disease a fungal disease and recommends copper and sulfur fungicides. To close that gap I condensed the corpus into a 2,200 character knowledge digest and embedded it in the model's GGUF chat template, in the branch that fires when no system message is supplied. Any runtime that renders the template injects it automatically. The same model loaded bare now names cassava mosaic disease correctly as a virus, cites the real resistant varieties Hope, Obasanjo-2, TME419, Dixon and Fineface, and gives correct green mite dosing.

Digest length was decisive. A first version of 5,677 characters failed completely, with the model falling back to generic advice about overwatering and never mentioning a cassava disease at all. Trimmed to roughly 2,200 characters, with symptom-to-diagnosis mappings placed first, it worked.

**Fine-tuning: evaluated and rejected.** I built 206 question-and-answer pairs from the corpus and trained a LoRA adapter on Llama-3.2-1B using peft and TRL, converting the merged model back to GGUF Q4_K_M. Training converged cleanly at a loss of 1.366 with mean token accuracy of 0.838. The resulting model was worse than the digest approach and I did not ship it. It garbled the variety yields and contradicted itself within a single answer, answered a weed question with disease content and invented a crop that does not exist, and most seriously answered an out-of-scope question with fabricated veterinary schedules despite a refusal for that exact question being present in its training data. Applying the digest on top of the fine-tuned model made it worse still, because fine-tuning at this data scale had damaged the model's instruction-following, so the digest could no longer steer it. The conclusion I draw is that at this scale, knowledge present at inference time beats knowledge diffused into weights.

---

## Tools

**llama.cpp** is required by the challenge and turned out to be the right constraint. It runs GGUF models on CPU with no Python machine learning stack at inference, which is what makes the offline requirement achievable rather than aspirational.

**llama-embedding rather than sentence-transformers.** This was forced on me and improved the design. My development machine is a 2015 Intel Mac, and PyTorch stopped releasing Intel Mac builds after version 2.2.2, which current sentence-transformers requires. Rather than treat that as a blocker, I moved embedding to llama.cpp itself. The result has no PyTorch dependency at all, which is lighter and more likely to install cleanly on the target machine.

**all-MiniLM-L6-v2 at F16** rather than a quantized embedder. The generator is aggressively quantized because it is large and the task is forgiving. The embedder is 22 million parameters, so full precision costs about 70 megabytes, and every retrieval depends on its output. Quantize where the model is large and the task tolerates it; keep precision where the model is small and the task is foundational.

**numpy** for similarity search rather than a vector database. With 688 passages the store is a few megabytes held in memory and searched with a single matrix multiplication. Chroma or FAISS would add dependencies and startup cost for no measurable gain at this scale.

**FastAPI** for the interface, serving one HTML page with no external assets. No CDN, no web fonts, no external scripts, because a page that fetches a stylesheet is not offline.

**A hand-written vocabulary map** rather than a query-rewriting model. Farmers describe symptoms in everyday language while guides use technical terms, and that gap broke retrieval repeatedly. A second model call would double response time on a machine where responses already take forty seconds. A dictionary runs instantly, cannot hallucinate, is auditable by an agronomist, and handles Igbo terms the same way it handles English ones.

---

## Benchmarks

Measured on a 2015 MacBook Air, Intel Core i5-5250U dual-core, 8 GB RAM, no discrete GPU. This is at or below the standard evaluation profile of 4 vCPU and 8 GB.

| Metric | Value |
|---|---|
| Model | cassava-advisor-1B-Q4_K_M, 771 MB |
| Generation speed | 18.06 tokens/second |
| Peak RSS | 1,446.17 MB |
| Steady state RSS | 1,381 MB |
| Time to first token | 9,169 ms |
| Thermal throttling | None observed |
| S_perf | 100 (capped at the 15 t/s reference) |
| S_eff | 79.8 |

Peak memory is 20 percent of the 7 GB budget. Under live use with the model server, the web application and the embedding process running together, sampled peak was 1,533 MB, still under a quarter of the ceiling.

Two independent profiler runs were compared using the profiler's own variance tolerances. All four metrics passed with deltas under 0.3 percent against tolerances of plus or minus 15 percent for memory and 25 percent for throughput.

One measurement note. With the model server and web application running in the background, throughput measured 12.12 t/s rather than 18.06, so all reported figures were taken on an otherwise idle machine. Background load on a dual-core processor materially affects the benchmark.

Docker-based containment testing was attempted but could not complete on macOS, where the container cannot access hardware thermal sensors and the profiler's thermal sampler loops. Memory headroom was verified by direct sampling instead.

---

## Safety Architecture

A model this small will produce fluent, confident, wrong answers. That is not a defect I can prompt away, and I stopped trying. The system assumes the model will fail and puts deterministic code around it.

**A relevance gate.** If the best retrieved passage scores below 0.45, the model is never called. A question about goat vaccination is declined in under a second rather than answered from the model's own uncertain knowledge. Code cannot be talked out of this; a prompt instruction can.

**Question classification in code.** The model could not reliably choose between diagnosing a problem and giving steps for a task, so it defaulted to diagnosing, and answered "which fertilizer should I use" with "the most likely cause of the problem is". Classifying in Python and sending only the relevant instruction fixed it.

**Removal, not warning.** Any sentence recommending a chemical absent from the retrieved passages is deleted before display, replaced by a visible note explaining why. An earlier version only warned about such recommendations while still printing them, which is not a safeguard.

This matters because of a specific failure. At one point the system produced a numbered list recommending copper-based fungicides and chlorothalonil for cassava mosaic disease. Mosaic disease is a virus. None of the four retrieved passages mentioned any chemical; the model invented all four prescriptions because it had been asked what to do and had nothing to say. A farmer acting on that would have spent money they do not have on a treatment that cannot work, while their crop continued to fail.

The web interface streams tokens as they arrive and so cannot strip text before display. It sends the offending character offsets to the frontend, which strikes them through in place. An earlier version had the client re-derive the sentence split and match by string equality, which silently failed open whenever an answer truncated without terminal punctuation. Computing the offsets once on the server and passing them removed the possibility of disagreement.

---

## Scope

The system is built to answer, and tested against, the following:

- Diagnosis from symptoms: cassava mosaic disease, bacterial blight, brown streak, anthracnose, bud necrosis, root rot, leaf spots, green mite, mealybug, grasshopper, termites, vertebrate pests
- Management for each of the above, including which problems have no chemical cure
- Variety selection: eleven improved Nigerian varieties with yields, dry matter, disease resistance, and which to choose for garri, fufu, starch, drought, weed suppression or mechanisation
- Planting: timing, cutting selection, spacing, depth, land preparation
- Weed control: the critical period, herbicide options, and canopy-based suppression
- Yield and starch improvement, including low-cost options
- Out-of-scope questions, which are declined rather than answered

---

## Limitations

**The model reasons poorly and I cannot fix that.** It has misattributed scientific names, inverted comparative statements from its own sources, and contradicted itself within a single answer. I tested four models across two parameter scales and the pattern held. This is what a 1B model does, and the safety architecture exists because of it rather than in spite of it.

**Retrieval ranks by similarity, not by understanding.** A question asking what chemical to spray retrieves chemical-dense passages regardless of what those chemicals treat. This produced answers recommending spider mite sprays for mealybug, where the chemicals were real and correctly sourced but applied to the wrong target. I mitigated it by rewriting the corpus so each control measure names its target explicitly and states which chemicals should not be used, which resolved all four adversarial cases I tested. The underlying limitation remains.

**Constraints in a question are largely ignored.** Asking how to increase yield without spending much money reliably retrieves fertiliser advice, because embedding models handle topics well and conditions poorly.

**It is not yet usable by a farmer directly.** Running the full system requires Python, llama.cpp and a terminal. The offline capability is real and demonstrated; the packaging is not built. It is currently a tool for an extension officer with a laptop, not for the farmer that officer visits.

**Igbo support is input only.** A farmer can describe symptoms in Igbo and retrieval works correctly, but the answer returns in English. Generating Igbo would need a model with real Igbo capability, and the small models that cover African languages either exclude Igbo or carry non-commercial licences.

**Answers take about forty seconds.** Acceptable for an extension officer, marginal for a farmer standing in a field.

---

## Roadmap

Everything below comes from the field conversations described above, not from my own guesses.

**Weed management.** The production manager at Green Hills named weed management as their single biggest problem, ahead of pests and disease. Several improved varieties in the IITA BASICS booklet, including Fineface, Obasanjo-2 and Baba-70, form a closing canopy that suppresses weeds by denying them light. That turns a variety choice into a weed control strategy, which reduces spraying cost rather than adding to it.

**Soil correction.** Green Hills had their fields assessed and found the soil acidic, with lime and poultry waste recommended. Deeper coverage of soil correction, particularly acidity in the humid south, would serve both smallholders and commercial operations.

**Voice.** The PhD agriculturist made the point that most farmers he meets do not read or write English. A text interface excludes exactly the people it is meant to serve. Speech recognition and audio output in Igbo would be the largest single step towards genuine reach, and it is why the vocabulary bridge already handles Igbo input.

**Photo identification.** The Green Hills manager asked directly whether he could photograph a weed or a soil sample and be told what it is. That matches how someone standing in a field actually wants to ask a question.

**Packaging.** A packaged installer that a farmer or extension officer can run without command line knowledge is the difference between a demonstrated capability and a usable tool.

---

## Sources

- IITA, Pest Control in Cassava Farms (IPM Field Guide)
- IITA, Disease Control in Cassava Farms (IPM Field Guide)
- IITA BASICS-II, Profiles of Best Performing Improved Cassava Varieties
- NAERLS, Cassava Production, Processing and Utilization
- Africa Soil Health Consortium, Cassava System Cropping Guide
- FAO, Save and Grow: Cassava

Cleaned source text is included in `corpus/` so the vector store can be rebuilt and verified.
