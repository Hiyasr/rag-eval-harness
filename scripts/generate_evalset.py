"""Generate an eval set for YOUR corpus with an LLM, keeping only questions whose
evidence quote verifiably exists in the chunk. Review a sample by hand before trusting it.
Usage: python scripts/generate_evalset.py --corpus path/to/docs --n 120 --out data/my_evalset.jsonl"""
import argparse, json, random, re
from ragkit.chunking import load_corpus
from ragkit.embed import tokenize
from ragkit.llm import AnthropicLLM

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()

SYS = ('Write ONE factual question answerable only from the passage. Reply JSON only: '
       '{"question": "...", "evidence": "<exact 4-12 word quote copied from the passage>", "reference": "<short answer>"}. '
       "The question must not name the passage or say 'according to the text'.")
llm, chunks = AnthropicLLM(), [c for c in load_corpus(a.corpus) if len(c.text) > 200]
random.seed(0); random.shuffle(chunks)
kept = []
for c in chunks:
    if len(kept) >= a.n:
        break
    try:
        d = json.loads(re.search(r"\{.*\}", llm.complete(SYS, c.text, 300), re.S).group(0))
        if " ".join(tokenize(d["evidence"])) in " ".join(tokenize(c.text)):
            kept.append({**d, "source": c.id})
    except Exception:
        continue
open(a.out, "w").write("\n".join(json.dumps(r) for r in kept) + "\n")
print(f"kept {len(kept)} verified questions -> {a.out}")
