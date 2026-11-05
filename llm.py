from __future__ import annotations
import os
import re
import time


class AnthropicLLM:
    """Thin wrapper with retries. Reads ANTHROPIC_API_KEY from the environment.
    Override the model with RAG_MODEL."""

    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("RAG_MODEL", "claude-haiku-5-5")

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        for attempt in range(4):
            try:
                r = self.client.messages.create(
                    model=self.model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in r.content if b.type == "text")
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""


class ExtractiveLLM:
    """Offline stand-in: returns the sentence of the top context with most query overlap.
    Used for tests/CI so the pipeline runs without an API key."""

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        q = re.search(r"QUESTION:\s*(.*)", user)
        qt = set(re.findall(r"[a-z0-9]+", (q.group(1) if q else "").lower()))
        ctx = re.findall(r"\[(\d+)\]\s(.*?)(?=\n\[\d+\]|\nQUESTION:)", user, flags=re.S)
        best, score = ("I don't know.", 0)
        for num, text in ctx:
            for s in re.split(r"(?<=[.!?])\s+", text):
                sc = len(qt & set(re.findall(r"[a-z0-9]+", s.lower())))
                if sc > score:
                    best, score = f"{s.strip()} [{num}]", sc
        return best
