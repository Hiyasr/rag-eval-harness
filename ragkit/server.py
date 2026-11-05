"""FastAPI service. Run: CORPUS=data/sample_corpus EMBEDDER=hash uvicorn ragkit.server:app"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from .chunking import load_corpus
from .embed import get_embedder
from .llm import AnthropicLLM, ExtractiveLLM
from .pipeline import RAGPipeline
from .retrieve import Retriever

app = FastAPI(title="ragkit")
_retriever = Retriever(load_corpus(os.environ.get("CORPUS", "data/sample_corpus")),
                       get_embedder(os.environ.get("EMBEDDER", "hash")), mode="hybrid")
_llm = ExtractiveLLM() if os.environ.get("OFFLINE") else AnthropicLLM()
_pipe = RAGPipeline(_retriever, _llm)


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"chunks": len(_retriever.chunks)}


@app.post("/ask")
def ask(body: Ask):
    a = _pipe.answer(body.question)
    return {"answer": a.text, "sources": a.chunk_ids, "latency_s": round(a.latency_s, 3)}
