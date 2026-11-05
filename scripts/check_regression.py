"""Fail (exit 1) if retrieval quality drops below thresholds.json. Used by CI."""
import json, sys
from ragkit.chunking import load_corpus
from ragkit.embed import get_embedder
from ragkit.retrieve import Retriever
from ragkit.evaluate import load_evalset, retrieval_metrics

th = json.load(open("thresholds.json"))
r = Retriever(load_corpus("data/sample_corpus", "paragraph"), get_embedder("hash"), mode="hybrid")
m = retrieval_metrics(r, load_evalset("data/evalset.jsonl"), 5)
print(m)
bad = {k: (m[k], t) for k, t in th.items() if m[k] < t}
if bad:
    print("REGRESSION:", bad)
    sys.exit(1)
print("quality gate passed")
