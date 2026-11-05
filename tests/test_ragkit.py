from ragkit.chunking import fixed_chunks, paragraph_chunks, load_corpus
from ragkit.embed import HashEmbedder
from ragkit.retrieve import BM25, Retriever, OverlapReranker
from ragkit.embed import tokenize
from ragkit.evaluate import (load_evalset, evidence_rank, retrieval_metrics, token_f1,
                             lexical_faithfulness, end_to_end)
from ragkit.pipeline import RAGPipeline
from ragkit.llm import ExtractiveLLM

CORPUS, EVAL = "data/sample_corpus", "data/evalset.jsonl"


def test_fixed_chunks_overlap_and_bounds():
    ch = fixed_chunks("d", "x" * 1200, size=500, overlap=100)
    assert all(len(c.text) <= 500 for c in ch) and len(ch) == 3


def test_paragraph_chunks_keep_heading_context():
    ch = paragraph_chunks("d", "# Title\n\nHello world.\n\nSecond para.")
    assert ch and ch[0].text.startswith("[Title]")


def test_bm25_prefers_matching_doc():
    docs = [tokenize("cats sleep all day"), tokenize("dogs bark at night")]
    s = BM25(docs).scores(tokenize("dogs bark"))
    assert s[1] > s[0]


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder().encode(["hello world", "hello world"])
    assert (e[0] == e[1]).all() and abs((e[0] ** 2).sum() - 1) < 1e-4


def test_evidence_rank():
    assert evidence_rank(["a b", "the quick brown fox"], "quick brown") == 2
    assert evidence_rank(["a b"], "zzz") is None


def test_metrics_helpers():
    assert token_f1("run pip install", "run pip install") == 1.0
    assert lexical_faithfulness("foo [1]", ["foo bar"]) == 1.0


def test_hybrid_beats_floor_on_sample_corpus():
    r = Retriever(load_corpus(CORPUS), HashEmbedder(), mode="hybrid")
    m = retrieval_metrics(r, load_evalset(EVAL), 5)
    assert m["recall@5"] >= 0.85 and m["mrr"] >= 0.6


def test_every_mode_runs_and_reranker_keeps_k():
    chunks = load_corpus(CORPUS)
    for mode in ("bm25", "dense", "hybrid"):
        r = Retriever(chunks, HashEmbedder(), mode=mode, reranker=OverlapReranker())
        assert len(r.search("How do I install Fluxo?", 3)) == 3


def test_end_to_end_offline():
    r = Retriever(load_corpus(CORPUS), HashEmbedder(), mode="hybrid")
    res = end_to_end(RAGPipeline(r, ExtractiveLLM()), load_evalset(EVAL)[:5])
    assert res["n"] == 5 and res["judge"] == "lexical-fallback" and res["latency_p95_s"] >= 0
