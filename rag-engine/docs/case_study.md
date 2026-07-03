# Case Study (draft — fill in after Phase 4/6)

## One-liner
"I built a RAG system with hybrid search that achieves **X%** faithfulness
and **Y%** citation accuracy on a 50-question eval suite."

## Sections to fill in
1. **The numbers, up front** — correctness / faithfulness / retrieval
   relevance / citation accuracy from `eval/run_eval.py`.
2. **Why hybrid beats dense-only for technical docs** — cite the
   dense-only vs. hybrid comparison from the dashboard toggle; technical
   docs contain exact-match tokens (function names, config keys, error
   codes) that dense embeddings can miss but BM25 catches.
3. **Chunking strategy comparison** — table from
   `eval/chunking_comparison.py`: which strategy won on which metric,
   and why (e.g. structure-aware chunking likely wins on retrieval
   relevance for well-headered docs; semantic chunking may win on
   faithfulness for prose-heavy docs).
4. **What "production maturity" means here** — citation verification,
   graceful "I don't know" handling, confidence scoring — and why most
   RAG tutorials skip these.
5. **Link to demo video** (<4 min, per Phase 6).
