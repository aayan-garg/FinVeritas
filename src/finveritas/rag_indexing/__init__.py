"""
FinVeritas RAG Indexing Module:
SEC filing document parsing (10-K, 10-Q, 8-K), financial table representation,
hybrid chunking, vector indexing, and retrieval.
"""

from finveritas.rag_indexing.chunking import (
    ChunkCorpusItem,
    build_chunk_corpus,
    chunk_text_sliding_window,
)
from finveritas.rag_indexing.evaluation import (
    QueryEvaluationResult,
    compute_ndcg_at_k,
    compute_paired_bootstrap,
    evaluate_retriever_on_financebench,
    is_numerical_question,
)
from finveritas.rag_indexing.financebench_data import (
    PageCorpusItem,
    build_page_corpus,
    download_file_with_retry,
    download_financebench_metadata,
    download_financebench_pdfs,
    extract_pdf_pages,
    load_financebench_metadata,
    load_page_corpus,
    normalize_financial_text,
    save_page_corpus,
)
from finveritas.rag_indexing.retrievers import (
    BM25Retriever,
    ChunkBM25Retriever,
    ChunkRetrievalResult,
    DenseRetriever,
    HybridNeuralRerankRetriever,
    HybridRRFRetriever,
    NeuralReranker,
    RetrievalResult,
    TfidfRetriever,
    deduplicate_chunks_to_pages,
    tokenize_financial_query,
)

__all__ = [
    "PageCorpusItem",
    "ChunkCorpusItem",
    "RetrievalResult",
    "ChunkRetrievalResult",
    "QueryEvaluationResult",
    "TfidfRetriever",
    "BM25Retriever",
    "ChunkBM25Retriever",
    "DenseRetriever",
    "HybridRRFRetriever",
    "NeuralReranker",
    "HybridNeuralRerankRetriever",
    "deduplicate_chunks_to_pages",
    "build_chunk_corpus",
    "chunk_text_sliding_window",
    "tokenize_financial_query",
    "normalize_financial_text",
    "download_file_with_retry",
    "download_financebench_metadata",
    "download_financebench_pdfs",
    "load_financebench_metadata",
    "extract_pdf_pages",
    "build_page_corpus",
    "save_page_corpus",
    "load_page_corpus",
    "evaluate_retriever_on_financebench",
    "compute_ndcg_at_k",
    "compute_paired_bootstrap",
    "is_numerical_question",
]

