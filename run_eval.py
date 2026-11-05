"""End-to-end eval (retrieval + generation + judge). Needs ANTHROPIC_API_KEY unless --offline.
Usage: python scripts/run_eval.py --corpus data/sample_corpus --evalset data/evalset.jsonl --embedder hash --offline"""
import argparse, json
from ragkit.chunking import load_corpus
from ragkit.embed import get_embedder
from ragkit.retrieve import Retriever, OverlapReranker, CrossEncoderReranker
from ragkit.pipeline import RAGPipeline
from ragkit.llm import AnthropicLLM, ExtractiveLLM
from ragkit.evaluate import load_evalset, end_to_end

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--evalset", required=True)
ap.add_argument("--embedder", default="hash")
ap.add_argument("--chunking", default="paragraph")
ap.add_argument("--mode", default="hybrid")
ap.add_argument("--reranker", default="none", choices=["none", "overlap", "cross-encoder"])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--offline", action="store_true", help="use extractive LLM + lexical judge (no API)")
ap.add_argument("--out", default="results/e2e.json")
a = ap.parse_args()

rr = {"none": None, "overlap": OverlapReranker(), "cross-encoder": None}[a.reranker]
if a.reranker == "cross-encoder":
    rr = CrossEncoderReranker()
r = Retriever(load_corpus(a.corpus, a.chunking), get_embedder(a.embedder), mode=a.mode, reranker=rr)
llm = ExtractiveLLM() if a.offline else AnthropicLLM()
judge = None if a.offline else AnthropicLLM()
res = end_to_end(RAGPipeline(r, llm, k=a.k), load_evalset(a.evalset), judge=judge, k=a.k)
json.dump(res, open(a.out, "w"), indent=2)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "rows"})
