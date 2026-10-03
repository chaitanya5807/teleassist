\# ROLE

You are a senior ML engineer. Build a complete, production-style, portfolio-grade project in this repository, end to end. Work phase by phase, commit after each phase, and run the tests and scripts yourself before moving on. Never invent results: every metric in the README must come from actually running the evaluation scripts.



\# ENVIRONMENT

\- I am on native Windows with PowerShell. `make` and bash are NOT available.

\- Provide PowerShell-friendly equivalents for every Makefile target and .sh script (for example a tasks.ps1, or plain `python -m` commands). You may still include the Makefile and .sh files for Linux/Colab use.

\- Use pathlib and avoid hard-coded path separators so the code runs on both Windows and Linux.

\- I have no local GPU. Anything needing a GPU (LoRA training, full generation eval) will be run later on a free Google Colab T4. Implement it fully, provide a CPU smoke-test path, and tell me exactly what to run on Colab.



\# PROJECT

"TeleAssist": a domain-specific RAG assistant for TELECOM customer support and consumer regulations (India-focused: tariffs, SIM/KYC, number portability, billing disputes, TRAI consumer rules, broadband/5G basics).

Core selling points (must be visible in code and README):

1\. SELF-BUILT retrieval stack (no LangChain/LlamaIndex retrievers): custom chunker, custom BM25, custom dense vector index, custom Reciprocal Rank Fusion, plus a cross-encoder reranker.

2\. LoRA/QLoRA fine-tune of a small open LLM so it answers in a grounded, cited, domain-appropriate style and abstains when evidence is missing.

3\. Rigorous evaluation with an ablation table comparing: Base LLM (no retrieval) vs RAG dense-only vs RAG BM25-only vs RAG hybrid vs RAG hybrid+rerank vs RAG hybrid+rerank + LoRA model.

4\. Deployed as FastAPI + Streamlit, containerized with Docker.



\# TECH STACK (use exactly these unless something is unavailable, then pick the closest and document why)

\- Python 3.11, PyTorch, transformers, peft, trl, bitsandbytes (optional, fall back to plain LoRA on CPU/no-GPU), datasets, accelerate

\- Embeddings: BAAI/bge-small-en-v1.5 (sentence-transformers)

\- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

\- Generator: Qwen/Qwen2.5-1.5B-Instruct (ungated, fits a free Colab T4). Make the model name a config value.

\- Vector index: own NumPy flat cosine index (normalized embeddings, matrix product + argpartition). Optionally add a FAISS backend behind the same interface purely for a speed comparison.

\- API: FastAPI + Pydantic v2. UI: Streamlit. Packaging: Docker + docker-compose. Tests: pytest. Lint: ruff. Config: YAML (configs/\*.yaml) loaded into Pydantic settings.

\- Reproducibility: fixed seeds, pinned requirements.txt, Makefile targets.



\# REPOSITORY STRUCTURE (create exactly this)

```

teleassist/

├── README.md

├── Makefile

├── tasks.ps1                 # PowerShell equivalent of Makefile targets

├── requirements.txt

├── pyproject.toml            # ruff config

├── Dockerfile

├── docker-compose.yml

├── .gitignore

├── configs/

│   ├── default.yaml          # chunk size, overlap, top\_k, models, weights

│   └── lora.yaml             # rank, alpha, dropout, target modules, lr, epochs

├── data/

│   ├── raw/                  # downloaded source docs (gitignored, with manifest)

│   ├── processed/            # cleaned text + chunks.jsonl

│   ├── train/                # train.jsonl, val.jsonl (LoRA SFT data)

│   └── eval/                 # eval\_set.jsonl (hand-verifiable test set)

├── src/teleassist/

│   ├── \_\_init\_\_.py

│   ├── config.py

│   ├── ingestion/

│   │   ├── download.py       # fetch public docs, write data/raw/MANIFEST.json (url, date, license note)

│   │   ├── parse.py          # PDF/HTML/MD -> clean text with metadata (source, page, section)

│   │   └── chunking.py       # custom recursive chunker w/ overlap, token-aware, keeps metadata

│   ├── retrieval/

│   │   ├── bm25.py           # BM25 (Okapi) implemented from scratch, tokenizer, inverted index, save/load

│   │   ├── dense.py          # embedder wrapper + custom flat vector index, save/load

│   │   ├── fusion.py         # Reciprocal Rank Fusion (+ weighted score fusion option)

│   │   ├── reranker.py       # cross-encoder reranking

│   │   └── pipeline.py       # Retriever class: mode in {bm25, dense, hybrid, hybrid\_rerank}

│   ├── generation/

│   │   ├── prompts.py        # grounded prompt template, citation format \[1]\[2], abstain rule

│   │   ├── llm.py            # load base or base+LoRA adapter, generate(), 4-bit optional

│   │   └── rag.py            # RAGAnswerer: retrieve -> prompt -> generate -> parse citations

│   ├── training/

│   │   ├── build\_sft\_data.py # create instruction data from chunks (see DATA section)

│   │   └── train\_lora.py     # QLoRA/LoRA SFT with TRL, logs loss, saves adapter to models/

│   ├── evaluation/

│   │   ├── metrics.py        # recall@k, MRR, hit-rate, token-F1, containment, citation accuracy, abstention rate

│   │   ├── judge.py          # optional LLM-as-judge faithfulness (pluggable, off by default)

│   │   ├── run\_retrieval\_eval.py

│   │   ├── run\_generation\_eval.py

│   │   └── make\_report.py    # builds results/summary.md + results/ablation.csv + plots

│   └── serving/

│       ├── api.py            # FastAPI: POST /ask, GET /health, GET /sources/{id}

│       └── app.py            # Streamlit chat UI showing answer, citations, retrieved chunks, latency

├── scripts/

│   ├── 01\_download.sh

│   ├── 02\_build\_index.sh

│   ├── 03\_build\_train\_data.sh

│   ├── 04\_train\_lora.sh

│   ├── 05\_evaluate.sh

│   ├── 06\_serve.sh

│   └── review\_eval\_set.py

├── notebooks/

│   └── colab\_train\_and\_eval.ipynb   # runs the whole pipeline on a free Colab T4

├── tests/

│   ├── test\_chunking.py

│   ├── test\_bm25.py              # compare ranking sanity vs a tiny hand-made corpus

│   ├── test\_dense\_index.py

│   ├── test\_fusion.py

│   ├── test\_metrics.py

│   └── test\_api.py

├── results/                       # generated by eval scripts only

├── models/                        # LoRA adapters (gitignored, README explains how to regenerate)

└── docs/

&#x20;   ├── architecture.md            # mermaid diagram + component explanations

&#x20;   └── design\_decisions.md        # why hybrid, chunk size, LoRA rank, with experiment evidence

```



\# DATA (do this carefully)

1\. Corpus: download 25-60 PUBLIC documents about Indian telecom consumer topics (e.g., TRAI consumer regulations and handbooks, TRAI MNP/DND/tariff/QoS rules, DoT KYC and SIM guidelines, public FAQ/help pages of telecom operators, plus Wikipedia articles on telecom concepts). Respect robots.txt and terms; store a MANIFEST.json with URL, retrieval date, and license note. If a source cannot be fetched, skip it and log it. Provide a small bundled fallback corpus (Wikipedia telecom pages via API) so the project still runs end to end.

2\. Split documents BY DOCUMENT into train / eval groups (about 80/20) so there is no leakage between LoRA training data and the evaluation set.

3\. SFT data (build\_sft\_data.py): for train-group chunks, generate question/answer pairs using a pluggable generator (OpenAI-compatible API via env var if present, otherwise a local instruct model). Each sample = {question, retrieved\_context (gold chunk plus 2-3 hard-negative chunks from BM25), answer with citations}. Also include about 15% UNANSWERABLE questions whose correct answer is "I could not find this in the provided documents." Target 1,500-3,000 samples. Filter low-quality pairs (length, answer must be supported by the gold chunk via token-overlap check).

4\. Eval set (eval\_set.jsonl): 80-100 questions from eval-group documents, fields: id, question, reference\_answer, gold\_chunk\_ids, answerable (bool), category. Include about 20% unanswerable and about 15% multi-chunk questions. Generate drafts automatically, then write a script `scripts/review\_eval\_set.py` that prints each item for quick human verification and marks verified items. Add a note in the README that the eval set was reviewed manually, and do NOT claim review happened unless the verified flag is set in the file.



\# RETRIEVAL DETAILS

\- Chunker: recursive split on headings, paragraphs, sentences; token-aware; configurable size (test 256, 512, 1024) and overlap (0, 64, 128); metadata preserved; stable chunk IDs.

\- BM25: implement Okapi BM25 yourself (k1=1.5, b=0.75 defaults, configurable), with lowercase, regex tokenization, stopword removal, and a simple stemmer (implement or use a tiny dependency). Inverted index serialized to disk.

\- Dense: batch-embed chunks, store float32 matrix + id map; cosine search via normalized dot product.

\- Fusion: RRF with k=60 over BM25 and dense top-50, then take top-20 to the reranker, return top-5 to the generator.

\- Reranker: cross-encoder scores query-chunk pairs; batch inference.

\- Run a chunk-size/overlap sweep and save results to results/chunk\_sweep.csv.



\# GENERATION DETAILS

\- Prompt: system instruction to answer ONLY from numbered context passages, cite as \[1], \[2], keep answers concise, and reply exactly "I could not find this in the provided documents." when the context lacks the answer.

\- LoRA config (configs/lora.yaml): r=16, alpha=32, dropout=0.05, target\_modules=\[q\_proj,k\_proj,v\_proj,o\_proj], lr=2e-4, 2-3 epochs, cosine schedule, 4-bit NF4 when CUDA is available. Log train/val loss and save curves to results/.

\- Provide a flag to run a tiny smoke-test training (20 steps) on CPU for CI.



\# EVALUATION (all numbers must be produced by code)

Retrieval metrics per mode (bm25, dense, hybrid, hybrid\_rerank): Recall@1/3/5/10, MRR@10, hit-rate@5.

Generation metrics for each of 6 systems: (A) base LLM no retrieval, (B) RAG dense, (C) RAG BM25, (D) RAG hybrid, (E) RAG hybrid+rerank, (F) RAG hybrid+rerank + LoRA:

\- token-F1 and answer containment vs reference

\- citation accuracy (cited passages contain the gold chunk)

\- abstention precision/recall on unanswerable questions

\- hallucination proxy: % of answers with unsupported claims (judge.py if enabled, otherwise lexical-overlap proxy, clearly labeled as a proxy)

\- mean and p95 latency per query

make\_report.py outputs results/ablation.csv, results/summary.md (markdown table), and PNG plots; the README embeds them. Include a short honest "Limitations" section (small eval set, synthetic training data, small model).



\# SERVING

\- FastAPI /ask returns {answer, citations\[{id, source, page, snippet}], retrieved\[...], latency\_ms, mode}. Mode selectable per request. Load models once at startup. Add request validation and basic error handling.

\- Streamlit UI: chat box, toggle retrieval mode and LoRA on/off, expandable "retrieved sources" panel, latency display.

\- Dockerfile (slim, CPU-capable) + docker-compose with api and ui services. Document GPU usage separately.



\# QUALITY BAR

\- Type hints and docstrings on public functions; ruff clean; pytest passing; no hard-coded paths (use config); logging instead of prints; clear error messages.

\- Makefile targets (and matching tasks.ps1 commands): setup, data, index, sft-data, train, eval, report, serve, test, lint, docker-build.

\- README.md must contain: one-paragraph pitch, architecture diagram (mermaid), quickstart (5 commands, for both PowerShell and Linux), results table pulled from results/summary.md, design decisions summary, repo structure, how to reproduce on free Colab, limitations, future work (query rewriting, multilingual Hindi support, RAGAS integration, vLLM serving).



\# WORKFLOW (follow strictly)

Phase 1 scaffold + config + tests skeleton -> Phase 2 ingestion + chunking -> Phase 3 BM25 + dense + fusion + reranker + retrieval eval -> Phase 4 RAG generation with base model -> Phase 5 SFT data build -> Phase 6 LoRA training -> Phase 7 full ablation eval + report -> Phase 8 API + UI + Docker -> Phase 9 README/docs polish.

After each phase: run tests, summarize what was done, list any deviations, and commit with a clear message. If a step needs a GPU or network that is unavailable, implement it fully, provide the smoke-test path, and state exactly what I must run manually (and where) to produce the real numbers. Do not fabricate outputs.

Do ONE phase at a time. Stop after each phase and wait for my approval before starting the next.



\# FINAL DELIVERABLE

At the end print: (1) the exact commands to run the full pipeline on a Colab T4, (2) a checklist of manual steps I still must do (review eval set, run training, run eval, paste results), and (3) 8 likely interview questions about this project with concise answers based on the actual code.

