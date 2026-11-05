"""Retrieval ablation: chunking x retrieval mode x reranker. No LLM calls needed.
Usage: python scripts/ablate.py --corpus data/sample_corpus --evalset data/evalset.jsonl --embedder hash
       python scripts/ablate.py --corpus <your docs> --evalset <yours> --embedder BAAI/bge-small-en-v1.5 --reranker cross-encoder"""
import argparse, itertools, json, time
from ragkit.chunking import load_corpus
from ragkit.embed import get_embedder
from ragkit.retrieve import Retriever, OverlapReranker, CrossEncoderReranker
from ragkit.evaluate import load_evalset, retrieval_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--evalset", required=True)
ap.add_argument("--embedder", default="hash")
ap.add_argument("--reranker", default="overlap", choices=["overlap", "cross-encoder"])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--out", default="results/ablation.md")
a = ap.parse_args()

emb = get_embedder(a.embedder)
rr = OverlapReranker() if a.reranker == "overlap" else CrossEncoderReranker()
ev = load_evalset(a.evalset)
rows = []
for chunking, mode, use_rr in itertools.product(["fixed", "paragraph"], ["bm25", "dense", "hybrid"], [False, True]):
    chunks = load_corpus(a.corpus, chunking)
    r = Retriever(chunks, emb, mode=mode, reranker=rr if use_rr else None)
    t0 = time.perf_counter()
    m = retrieval_metrics(r, ev, a.k)
    ms = (time.perf_counter() - t0) / len(ev) * 1000
    rows.append({"chunking": chunking, "mode": mode, "rerank": use_rr, "chunks": len(chunks), **m, "ms_per_query": round(ms, 2)})

rows.sort(key=lambda x: (-x[f"recall@{a.k}"], -x["mrr"]))
md = [f"Embedder: `{emb.name}` | reranker: `{rr.name}` | queries: {len(ev)}\n",
      f"| chunking | retrieval | rerank | chunks | recall@{a.k} | MRR | ms/query |", "|---|---|---|---|---|---|---|"]
for x in rows:
    md.append(f"| {x['chunking']} | {x['mode']} | {'yes' if x['rerank'] else 'no'} | {x['chunks']} | "
              f"{x[f'recall@{a.k}']:.3f} | {x['mrr']:.3f} | {x['ms_per_query']} |")
open(a.out, "w").write("\n".join(md) + "\n")
json.dump(rows, open(a.out.replace(".md", ".json"), "w"), indent=2)
print("\n".join(md))
