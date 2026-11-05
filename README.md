# ragkit — RAG with a measurable evaluation harness

Hybrid retrieval (BM25 + FAISS dense, fused with reciprocal-rank fusion), optional cross-encoder
reranking, cited answers, and an evaluation harness that reports **recall@k, MRR, faithfulness,
answer correctness and latency p50/p95**, plus an ablation over chunking × retrieval × reranking
and a CI quality gate that fails the build when retrieval quality regresses.

```
docs ──► chunking (fixed | paragraph) ──► BM25 index ─┐
                                          FAISS index ─┴─► RRF fusion ─► reranker ─► top-k
question ─────────────────────────────────────────────────────────────────────────┘     │
                                                                  LLM (cited answer) ◄───┘
```

## Quickstart (offline, no API key)
```bash
pip install -e ".[dev]"
pytest -q
python scripts/ablate.py --corpus data/sample_corpus --evalset data/evalset.jsonl --embedder hash
python scripts/check_regression.py
```

## Run it on a REAL corpus (this is what makes the project credible)
```bash
pip install -e ".[dense,dev]"
export ANTHROPIC_API_KEY=...            # optional: RAG_MODEL=<model id> to change model
python scripts/fetch_corpus.py https://github.com/<org>/<docs-heavy-repo> --subdir docs
python scripts/generate_evalset.py --corpus data/corpus --n 120 --out data/my_evalset.jsonl
#   -> hand-check ~20 generated questions before trusting the set
python scripts/ablate.py --corpus data/corpus --evalset data/my_evalset.jsonl \
       --embedder BAAI/bge-small-en-v1.5 --reranker cross-encoder
python scripts/run_eval.py --corpus data/corpus --evalset data/my_evalset.jsonl \
       --embedder BAAI/bge-small-en-v1.5 --reranker cross-encoder
```
Serve it: `CORPUS=data/corpus EMBEDDER=BAAI/bge-small-en-v1.5 uvicorn ragkit.server:app`
(or `docker build -t ragkit . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY ragkit`).

## Metrics
- **recall@k / MRR**: an eval row has an `evidence` quote; a hit is a retrieved chunk containing it.
- **faithfulness / correctness**: LLM judge (JSON verdicts) against context and reference.
  `--offline` swaps in a lexical fallback so CI runs without keys.
- **latency**: wall-clock p50/p95 per query (retrieval + generation).

## Results (fill in from YOUR run; do not copy the sample numbers)
| config | recall@5 | MRR | faithfulness | correctness | p95 latency |
|---|---|---|---|---|---|
| bm25 | | | | | |
| dense | | | | | |
| hybrid | | | | | |
| hybrid + rerank | | | | | |

The bundled `sample_corpus` (a fictional CLI's docs, 15 questions) only verifies the plumbing;
it is too small to separate the configurations.

## Limitations to state honestly in your write-up
- Evidence-string matching undercounts hits when the answer spans chunks.
- LLM-judge scores should be spot-checked against human labels on a sample.
- Generated eval questions skew toward single-chunk lookups.

> Status: code, tests and CI complete. Evaluation on a real corpus in progress; results tables will be filled from measured runs.

## CI
`.github/workflows/ci.yml` runs tests and `scripts/check_regression.py` against `thresholds.json`.
