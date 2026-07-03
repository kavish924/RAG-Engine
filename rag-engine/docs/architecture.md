# Architecture

## Data flow

```
Docs (md/txt/html/pdf)
  -> loaders.py (normalize + metadata)
  -> chunkers/{fixed_size,recursive_structure,semantic}.py
  -> embeddings.py (text-embedding-3-small)
  -> dedup.py (cosine > 0.95 -> skip)
  -> storage: vector_store.py (ChromaDB) + bm25_store.py, kept in sync

Query
  -> retrieval/dense.py            (top-k cosine)
  -> retrieval/sparse_bm25.py      (top-k BM25)
  -> retrieval/fusion.py           (RRF, weighted)
  -> retrieval/reranker.py         (top-20 -> top-5)
  -> generation/generator.py       (grounded answer + raw citations)
  -> generation/citation_verifier.py (per-claim verification)
  -> generation/confidence_scorer.py (composite score)
  -> if confidence < threshold: structured fallback ("I don't know")
  -> API response (app/api/routes/ask.py)
```

## Key design decisions

- **Two indexes, kept in sync.** Every chunk written to ChromaDB is also
  added to the BM25 index at the same time (`app/ingestion/pipeline.py`).
- **Chunking strategy is chunk-level metadata**, not a fixed choice — this
  lets `eval/chunking_comparison.py` run the same eval suite across all
  three strategies without re-architecting anything.
- **Citations are verified, not trusted.** The generator's citations are
  treated as a hypothesis; `citation_verifier.py` checks each one against
  the actual chunk content before it's shown to the user or counted in
  the confidence score.

(Fill in with real numbers once eval/chunking_comparison.py has run.)
