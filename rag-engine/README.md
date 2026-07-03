# RAG Engine — Hybrid Retrieval, Citation-Verified Q&A

A production-oriented Retrieval-Augmented Generation system with hybrid
(dense + sparse) retrieval, configurable chunking strategies, citation
verification, confidence scoring, and a full evaluation harness.

## Project Structure

```
rag-engine/
├── app/                        # Core application (FastAPI service)
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py                # Settings (env vars, model names, etc.)
│   ├── api/                    # Phase 5 — HTTP layer
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── routes/
│   │       ├── ask.py            # POST /v1/ask
│   │       ├── documents.py      # GET  /v1/documents
│   │       └── ingest.py         # POST /v1/ingest
│   ├── ingestion/               # Phase 1 — load, chunk, embed, dedup
│   │   ├── loaders.py            # multi-format doc loader (md/txt/html/pdf)
│   │   ├── chunkers/
│   │   │   ├── fixed_size.py         # baseline: fixed-size + overlap
│   │   │   ├── recursive_structure.py # structure-aware: split by headers
│   │   │   └── semantic.py           # semantic: split on topic boundaries
│   │   ├── embeddings.py         # text-embedding-3-small wrapper
│   │   ├── dedup.py              # near-duplicate detection (cosine > 0.95)
│   │   └── pipeline.py           # orchestrates the full ingestion flow
│   ├── retrieval/               # Phase 2 — hybrid retrieval engine
│   │   ├── dense.py               # vector search (ChromaDB)
│   │   ├── sparse_bm25.py         # BM25 keyword search
│   │   ├── fusion.py              # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py            # cross-encoder / LLM-as-judge rerank
│   │   └── retriever.py           # orchestrates dense+sparse+fusion+rerank
│   ├── generation/              # Phase 3 — grounded generation + trust layer
│   │   ├── prompts.py             # grounded system prompt templates
│   │   ├── generator.py           # calls the LLM, parses citations
│   │   ├── citation_verifier.py   # verifies each citation-claim pair
│   │   └── confidence_scorer.py   # composite confidence score
│   └── storage/                 # Vector store + BM25 index wrappers
│       ├── vector_store.py
│       └── bm25_store.py
├── eval/                        # Phase 4 — evaluation framework
│   ├── golden_dataset.jsonl      # 50+ hand-written Q&A pairs
│   ├── metrics/
│   │   ├── correctness.py         # LLM-as-judge vs. golden answer
│   │   ├── faithfulness.py        # are claims grounded in context?
│   │   ├── retrieval_relevance.py # were the right chunks retrieved?
│   │   └── citation_accuracy.py   # do citations support claims?
│   ├── run_eval.py               # runs full suite, produces report
│   └── chunking_comparison.py    # compares the 3 chunking strategies
├── frontend/                    # Phase 5 — query dashboard
│   └── streamlit_app.py          # ask questions, see citations/confidence,
│                                  # hybrid vs dense-only toggle
├── scripts/
│   └── seed_corpus.py            # indexes sample_corpus/ for reviewers
│   └── sample_corpus/            # sample documentation set
├── tests/                       # unit tests per module
├── docs/                        # Phase 6 — portfolio polish
│   ├── architecture.md
│   └── case_study.md             # "X% faithfulness, Y% citation accuracy"
├── docker-compose.yml            # API + ChromaDB + frontend
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Build Order (matches the phase plan)

| Phase | Days | Folder(s) |
|-------|------|-----------|
| 1. Ingestion & Chunking | 1–3 | `app/ingestion/` |
| 2. Hybrid Retrieval | 3–6 | `app/retrieval/` |
| 3. Generation & Citations | 6–9 | `app/generation/` |
| 4. Evaluation Framework | 9–11 | `eval/` |
| 5. API & Dashboard | 11–13 | `app/api/`, `frontend/`, `docker-compose.yml` |
| 6. Portfolio Polish | 13–14 | `docs/` + demo video (not in repo) |

## Quickstart

```bash
cp .env.example .env          # add your OPENAI_API_KEY / ANTHROPIC_API_KEY
docker-compose up --build     # spins up API + ChromaDB + dashboard
python scripts/seed_corpus.py # indexes the sample corpus
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

## Running Evals

```bash
python eval/run_eval.py                 # full metric suite
python eval/chunking_comparison.py       # compares fixed/recursive/semantic
```
