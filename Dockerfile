FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[dense]"
ENV CORPUS=data/sample_corpus EMBEDDER=BAAI/bge-small-en-v1.5
CMD ["uvicorn", "ragkit.server:app", "--host", "0.0.0.0", "--port", "8000"]
