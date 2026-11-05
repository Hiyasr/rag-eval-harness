# RAGkit: RAG with a measurable evaluation harness

[![CI](https://github.com/Hiyasr/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Hiyasr/rag-eval-harness/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**A retrieval-augmented generation (RAG) system built to be measured, not just demoed.**
Hybrid retrieval (BM25 + FAISS, fused with reciprocal-rank fusion), optional cross-encoder reranking and
cited answers, plus an evaluation harness and a CI quality gate that fails the build when retrieval
quality regresses.

> **Status:** code, tests and CI are complete. Evaluation on a real documentation corpus is in progress;
> the real-corpus results table below is filled in only from measured runs.

## What it does

```
docs ──► chunking (fixed | paragraph) ──► BM25 index ──┐
                                          FAISS index ──┴─► RRF fusion ─► reranker ─► top-k
question ──────────────────────────────────────────────────────────────────────────────┘ │
                                                           LLM (cited answer) ◄──────────┘
```

| Component | Options |
|---|---|
| Chunking | fixed-size with overlap, or paragraph-aware (keeps the nearest heading as context) |
| Retrieval | BM25, dense (FAISS), or hybrid with reciprocal-rank fusion |
| Reranking | none, lexical overlap (offline), or cross-encoder |
| Generation | Anthropic API, answers cite passages as `[1]`, `[2]` and say "I don't know" when the context lacks the answer |
| Evaluation | recall@k, MRR, faithfulness, correctness, latency p50/p95 |

Everything runs **offline** with a deterministic hash embedder and an extractive stand-in for the LLM,
so tests and CI need no API key or model download.

## Quickstart (offline, no API key)

```bash
pip install -e ".[dev]"
pytest -q
python scripts/ablate.py --corpus data/sample_corpus --evalset data/evalset.jsonl --embedder hash
python scripts/check_regression.py
```

## Run on a real corpus

```bash
pip install -e ".[dense,dev]"
export ANTHROPIC_API_KEY=...        # optional: RAG_MODEL=<model id> to pick the model

# 1. fetch docs from any documentation-heavy repo
python scripts/fetch_corpus.py https://github.com/<org>/<repo> --subdir docs

# 2. generate an eval set (only questions whose evidence quote exists in the source are kept)
python scripts/generate_evalset.py --corpus data/corpus --n 120 --out data/my_evalset.jsonl
#    then hand-check ~20 rows before trusting it

# 3. ablation: chunking x retrieval x reranking (retrieval metrics only, no LLM calls)
python scripts/ablate.py --corpus data/corpus --evalset data/my_evalset.jsonl \
    --embedder BAAI/bge-small-en-v1.5 --reranker cross-encoder

# 4. end-to-end evaluation: retrieval + generation + judge
python scripts/run_eval.py --corpus data/corpus --evalset data/my_evalset.jsonl \
    --embedder BAAI/bge-small-en-v1.5 --reranker cross-encoder
```

Serve it as an API:

```bash
CORPUS=data/corpus EMBEDDER=BAAI/bge-small-en-v1.5 uvicorn ragkit.server:app
# or
docker build -t ragkit . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY ragkit
```

`POST /ask {"question": "..."}` returns the answer, source chunk ids and latency.

## Metrics

| Metric | Definition |
|---|---|
| **recall@k** | share of questions where a top-k chunk contains the gold evidence quote |
| **MRR** | mean reciprocal rank of the first chunk containing the evidence |
| **faithfulness** | share of answers whose claims are all supported by the retrieved context (LLM judge) |
| **correctness** | share of answers that agree with the reference answer in substance (LLM judge) |
| **latency** | wall-clock p50 / p95 per query, retrieval + generation |

`--offline` swaps the LLM judge for a lexical fallback so CI runs without keys. Judge type is recorded in every result file.

## Results

### Real corpus

_In progress. Corpus, eval-set size and model will be listed here once the runs are complete._

| config | recall@5 | MRR | faithfulness | correctness | p95 latency |
|---|---|---|---|---|---|
| bm25 | TBD | TBD | TBD | TBD | TBD |
| dense | TBD | TBD | TBD | TBD | TBD |
| hybrid | TBD | TBD | TBD | TBD | TBD |
| hybrid + rerank | TBD | TBD | TBD | TBD | TBD |

### Pipeline sanity check (not a finding)

Measured on the bundled sample corpus (docs for a fictional CLI, 15 questions, hash embedder,
paragraph chunking, no reranker). This only confirms the plumbing works; the corpus is far too small
to rank the configurations.

| retrieval | recall@5 | MRR |
|---|---|---|
| bm25 | 1.000 | 0.889 |
| dense | 0.867 | 0.719 |
| hybrid | 0.933 | 0.806 |

## Failure analysis

_To be written after the real-corpus runs: a categorized breakdown of the questions each configuration
gets wrong (chunk boundary, vocabulary mismatch, multi-chunk answer, judge disagreement)._

## Limitations

- Evidence-string matching undercounts hits when an answer spans more than one chunk.
- LLM-judge scores should be spot-checked against human labels; the disagreement rate will be reported with the results.
- Generated eval questions skew toward single-chunk lookups, so a hand-written multi-hop subset is added.
- The hash embedder and extractive LLM are test doubles, not quality baselines.

## Project layout

```
ragkit/            chunking, embedders, BM25 + FAISS retrieval, reranking, pipeline, evaluation, API server
scripts/           fetch_corpus, generate_evalset, ablate, run_eval, check_regression
data/              sample_corpus + evalset.jsonl (offline tests); put real corpora in data/corpus (git-ignored)
tests/             unit tests for chunking, retrieval, metrics and the offline pipeline
thresholds.json    minimum recall@5 / MRR enforced by CI
```

## CI

`.github/workflows/ci.yml` runs the test suite and `scripts/check_regression.py`, which fails the build if
recall@5 or MRR drop below the values in `thresholds.json`.
