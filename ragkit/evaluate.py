"""Metrics: retrieval recall@k / MRR (evidence-string match), faithfulness and correctness
(LLM judge, with a lexical fallback), latency percentiles."""
from __future__ import annotations
import json
import re
import statistics
from pathlib import Path
from .embed import tokenize


def load_evalset(path: str | Path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        assert {"question", "evidence", "reference"} <= r.keys(), f"bad row: {r}"
    return rows


def _norm(s: str) -> str:
    return " ".join(tokenize(s))


def evidence_rank(contexts: list[str], evidence: str) -> int | None:
    """1-based rank of the first retrieved chunk containing the evidence string."""
    ev = _norm(evidence)
    for i, c in enumerate(contexts):
        if ev in _norm(c):
            return i + 1
    return None


def retrieval_metrics(retriever, evalset: list[dict], k: int = 5) -> dict:
    ranks = []
    for row in evalset:
        got = retriever.search(row["question"], k)
        ranks.append(evidence_rank([c.text for c in got], row["evidence"]))
    hits = [r for r in ranks if r]
    return {
        f"recall@{k}": len(hits) / len(ranks),
        "mrr": sum(1 / r for r in hits) / len(ranks),
        "n": len(ranks),
    }


def token_f1(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def lexical_faithfulness(answer: str, contexts: list[str]) -> float:
    clean = re.sub(r"\[\d+\]", "", answer)
    a = set(tokenize(clean))
    if not a:
        return 0.0
    ctx = set(tokenize(" ".join(contexts)))
    return len(a & ctx) / len(a)


JUDGE_SYS = ("You are a strict grader. Reply with JSON only: "
             '{"faithful": true|false, "correct": true|false}. '
             "faithful = every claim in ANSWER is supported by CONTEXT. "
             "correct = ANSWER agrees with REFERENCE in substance.")


def llm_judge(judge, question: str, answer: str, contexts: list[str], reference: str) -> dict:
    prompt = (f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}\n"
              "CONTEXT:\n" + "\n".join(contexts))
    raw = judge.complete(JUDGE_SYS, prompt, max_tokens=60)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        d = json.loads(m.group(0))
        return {"faithful": bool(d["faithful"]), "correct": bool(d["correct"])}
    except Exception:
        return {"faithful": None, "correct": None}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def end_to_end(pipeline, evalset: list[dict], judge=None, k: int = 5) -> dict:
    rows, lat = [], []
    for row in evalset:
        a = pipeline.answer(row["question"])
        lat.append(a.latency_s)
        rank = evidence_rank(a.contexts, row["evidence"])
        if judge is not None:
            j = llm_judge(judge, row["question"], a.text, a.contexts, row["reference"])
        else:
            j = {"faithful": lexical_faithfulness(a.text, a.contexts) >= 0.8,
                 "correct": token_f1(a.text, row["reference"]) >= 0.5}
        rows.append({"question": row["question"], "answer": a.text, "hit_rank": rank, **j})
    scored = [r for r in rows if r["faithful"] is not None]
    n = len(rows)
    return {
        f"recall@{k}": sum(1 for r in rows if r["hit_rank"]) / n,
        "faithfulness": sum(1 for r in scored if r["faithful"]) / max(len(scored), 1),
        "correctness": sum(1 for r in scored if r["correct"]) / max(len(scored), 1),
        "latency_p50_s": statistics.median(lat),
        "latency_p95_s": pct(lat, 0.95),
        "judge": "llm" if judge is not None else "lexical-fallback",
        "n": n,
        "rows": rows,
    }
