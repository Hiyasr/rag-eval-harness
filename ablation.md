Embedder: `hash-512` | reranker: `overlap` | queries: 15

| chunking | retrieval | rerank | chunks | recall@5 | MRR | ms/query |
|---|---|---|---|---|---|---|
| fixed | bm25 | no | 10 | 1.000 | 1.000 | 0.09 |
| fixed | bm25 | yes | 10 | 1.000 | 0.967 | 0.31 |
| fixed | hybrid | no | 10 | 1.000 | 0.967 | 0.19 |
| fixed | hybrid | yes | 10 | 1.000 | 0.933 | 0.47 |
| fixed | dense | yes | 10 | 1.000 | 0.917 | 0.36 |
| paragraph | bm25 | no | 20 | 1.000 | 0.889 | 0.11 |
| paragraph | bm25 | yes | 20 | 1.000 | 0.889 | 0.28 |
| fixed | dense | no | 10 | 1.000 | 0.883 | 0.12 |
| paragraph | dense | yes | 20 | 1.000 | 0.878 | 0.35 |
| paragraph | hybrid | yes | 20 | 1.000 | 0.867 | 0.48 |
| paragraph | hybrid | no | 20 | 0.933 | 0.806 | 0.23 |
| paragraph | dense | no | 20 | 0.867 | 0.719 | 0.11 |
