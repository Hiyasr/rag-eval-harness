from __future__ import annotations
import time
from dataclasses import dataclass, field
from .retrieve import Retriever

SYSTEM = (
    "Answer the question using ONLY the numbered context passages. "
    "Cite passages like [1] after each claim. If the context does not contain the answer, "
    "reply exactly: I don't know."
)


@dataclass
class Answer:
    question: str
    text: str
    contexts: list[str]
    chunk_ids: list[str]
    latency_s: float
    retrieval_s: float = 0.0
    meta: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm, k: int = 5):
        self.retriever, self.llm, self.k = retriever, llm, k

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        chunks = self.retriever.search(question, self.k)
        t1 = time.perf_counter()
        ctx = "\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
        text = self.llm.complete(SYSTEM, f"{ctx}\nQUESTION: {question}")
        t2 = time.perf_counter()
        return Answer(question, text.strip(), [c.text for c in chunks], [c.id for c in chunks],
                      latency_s=t2 - t0, retrieval_s=t1 - t0)
