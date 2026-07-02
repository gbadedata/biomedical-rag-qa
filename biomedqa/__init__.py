"""biomedqa: a small, reusable retrieval-augmented QA toolkit for biomedical literature.

Pipeline: ingest -> corpus (chunk) -> retrieve (BM25 / dense-FAISS) -> generate (LLM) -> evaluate.
Each stage is importable and independently testable.
"""
__version__ = "0.1.0"
