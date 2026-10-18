"""
FinVeritas RAG Indexing Module:
SEC filing document parsing (10-K, 10-Q, 8-K), financial table representation,
hybrid chunking, vector indexing, and retrieval.
"""

from finveritas.rag_indexing.evaluation import (
    QueryEvaluationResult,
    evaluate_retriever_on_financebench,
)
from finveritas.rag_indexing.financebench_data import (
    PageCorpusItem,
    build_page_corpus,
    download_file_with_retry,
    download_financebench_metadata,
    download_financebench_pdfs,
    extract_pdf_pages,
    load_financebench_metadata,
    normalize_financial_text,
)
from finveritas.rag_indexing.retrievers import (
    BM25Retriever,
    RetrievalResult,
    TfidfRetriever,
    tokenize_financial_query,
)

__all__ = [
    "PageCorpusItem",
    "RetrievalResult",
    "QueryEvaluationResult",
    "TfidfRetriever",
    "BM25Retriever",
    "tokenize_financial_query",
    "normalize_financial_text",
    "download_file_with_retry",
    "download_financebench_metadata",
    "download_financebench_pdfs",
    "load_financebench_metadata",
    "extract_pdf_pages",
    "build_page_corpus",
    "evaluate_retriever_on_financebench",
]
