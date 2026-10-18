"""
FinVeritas Unit Tests: Advanced Hybrid Financial Retrieval Engine.

Covers:
  - Chunk sliding window tokenization & financial symbol preservation
  - Chunk-to-page mapping and duplicate-page elimination
  - ChunkBM25 ranking and deterministic tie-breaking
  - Reciprocal Rank Fusion (RRF k=60) mathematical properties
  - Binary relevance nDCG@K edge cases & mathematical correctness
  - Multi-evidence annotation tracking & coverage
  - Paired cluster bootstrap resampling properties
  - Embedding cache key invalidation on content/model change
  - Numerical vs prose question classification
  - Missing document coverage and graceful fallback
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

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
from finveritas.rag_indexing.financebench_data import PageCorpusItem
from finveritas.rag_indexing.retrievers import (
    ChunkBM25Retriever,
    ChunkRetrievalResult,
    DenseRetriever,
    HybridRRFRetriever,
    deduplicate_chunks_to_pages,
)


def test_chunk_sliding_window_preserves_financial_notation():
    """
    Verify sliding window chunking preserves accounting brackets, signs, currencies, and
    percentages.
    """
    text = (
        "In FY2022, 3M reported net sales of $34,229 million, down (3.2)% from 2021. "
        "Operating cash flows were $5,579M with capital expenditures of $(1,750) million. "
        "The effective tax rate was 18.2% compared to 17.8% in the prior period."
    )
    chunks = chunk_text_sliding_window(
        text=text,
        doc_name="3M_2022_10K",
        page_index=42,
        chunk_size=20,
        chunk_overlap=5,
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_name == "3M_2022_10K"
        assert c.page_index == 42
        assert c.chunk_id.startswith("3M_2022_10K_p42_c")
        assert c.token_count > 0

    combined = " ".join(c.text for c in chunks)
    assert "$34,229" in combined
    assert "(3.2)%" in combined or "3.2%" in combined
    assert "$(1,750)" in combined or "1,750" in combined
    assert "18.2%" in combined


def test_build_chunk_corpus_aggregates_correctly():
    """Verify build_chunk_corpus processes multiple pages into an indexed chunk list."""
    pages = [
        PageCorpusItem(
            page_id="DOC1_p0",
            doc_name="DOC1",
            page_index=0,
            text="Revenue in Q1 was $100M. Operating expenses were $60M.",
            doc_link="",
            char_count=52,
        ),
        PageCorpusItem(
            page_id="DOC1_p1",
            doc_name="DOC1",
            page_index=1,
            text="Net income in Q1 was $40M, representing a 40% margin.",
            doc_link="",
            char_count=53,
        ),
    ]
    chunks, stats = build_chunk_corpus(pages, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 2
    assert stats["input_pages"] == 2
    assert stats["total_chunks_created"] == 2
    assert stats["avg_chunks_per_page"] == 1.0


def test_deduplicate_chunks_to_pages_collapsing():
    """Verify multiple chunks from the same page collapse into a single highest-ranked page."""
    chunks = [
        ChunkRetrievalResult(
            chunk_id="AMZN_p10_c1",
            doc_name="AMZN",
            page_index=10,
            chunk_index=1,
            score=9.5,
            rank=1,
            text_snippet="AWS revenue was $20B.",
        ),
        ChunkRetrievalResult(
            chunk_id="MSFT_p5_c0",
            doc_name="MSFT",
            page_index=5,
            chunk_index=0,
            score=8.2,
            rank=2,
            text_snippet="Azure cloud grew 30%.",
        ),
        ChunkRetrievalResult(
            chunk_id="AMZN_p10_c0",  # Duplicate page (rank 3, lower score)
            doc_name="AMZN",
            page_index=10,
            chunk_index=0,
            score=7.8,
            rank=3,
            text_snippet="Amazon segments include AWS and Retail.",
        ),
        ChunkRetrievalResult(
            chunk_id="AAPL_p20_c0",
            doc_name="AAPL",
            page_index=20,
            chunk_index=0,
            score=6.0,
            rank=4,
            text_snippet="iPhone sales hit records.",
        ),
    ]
    pages = deduplicate_chunks_to_pages(chunks, top_k=3)
    assert len(pages) == 3
    # First should be AMZN page 10
    assert pages[0].doc_name == "AMZN" and pages[0].page_index == 10
    assert pages[0].rank == 1
    assert pages[0].score == 9.5

    # Second should be MSFT page 5
    assert pages[1].doc_name == "MSFT" and pages[1].page_index == 5
    assert pages[1].rank == 2

    # Third should be AAPL page 20 (duplicate AMZN page 10 skipped)
    assert pages[2].doc_name == "AAPL" and pages[2].page_index == 20
    assert pages[2].rank == 3


def test_chunk_bm25_retrieval_and_ties():
    """Verify ChunkBM25Retriever scores terms and preserves deterministic ranking."""
    chunks = [
        ChunkCorpusItem(
            chunk_id="c0",
            doc_name="DOC_A",
            page_index=0,
            chunk_index=0,
            text="capital expenditure capex additions to property and equipment",
            char_start=0,
            char_end=60,
            token_count=8,
        ),
        ChunkCorpusItem(
            chunk_id="c1",
            doc_name="DOC_A",
            page_index=1,
            chunk_index=0,
            text="research and development expenses operating costs",
            char_start=0,
            char_end=50,
            token_count=6,
        ),
        ChunkCorpusItem(
            chunk_id="c2",
            doc_name="DOC_B",
            page_index=0,
            chunk_index=0,
            text="unrelated employee severance benefits restructuring charges",
            char_start=0,
            char_end=58,
            token_count=6,
        ),
    ]
    retriever = ChunkBM25Retriever().fit(chunks)
    res = retriever.retrieve_chunks("What was the capital expenditure capex?", top_k=2)
    assert len(res) >= 1
    assert res[0].chunk_id == "c0"
    assert res[0].score > 0.0

    # Test page deduplication wrapper
    page_res = retriever.retrieve("capital expenditure capex", top_k=2)
    assert len(page_res) >= 1
    assert page_res[0].doc_name == "DOC_A" and page_res[0].page_index == 0


def test_hybrid_rrf_scoring_math():
    """Verify Reciprocal Rank Fusion computes exact 1/(k + rank) sums with k=60."""
    bm25_mock = MagicMock()
    dense_mock = MagicMock()

    # Query: chunk_1 is rank 1 in BM25, rank 2 in Dense
    # Query: chunk_2 is rank 2 in BM25, rank 1 in Dense
    bm25_mock.retrieve_chunks.return_value = [
        ChunkRetrievalResult("c1", "DOC", 0, 0, 10.0, 1, "text1"),
        ChunkRetrievalResult("c2", "DOC", 1, 0, 8.0, 2, "text2"),
    ]
    dense_mock.retrieve_chunks.return_value = [
        ChunkRetrievalResult("c2", "DOC", 1, 0, 0.95, 1, "text2"),
        ChunkRetrievalResult("c1", "DOC", 0, 0, 0.90, 2, "text1"),
    ]

    rrf = HybridRRFRetriever(bm25_mock, dense_mock, rrf_k=60)
    fused = rrf.retrieve_chunks("query", top_k=2)
    assert len(fused) == 2

    # Both c1 and c2 score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0325224
    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert math.isclose(fused[0].score, expected_score, rel_tol=1e-5)
    assert math.isclose(fused[1].score, expected_score, rel_tol=1e-5)


def test_ndcg_at_k_computation_edge_cases():
    """Verify nDCG@K under binary relevance across perfect, partial, and empty rankings."""
    gold = {("DOC_A", 1), ("DOC_A", 2)}

    # Perfect ranking: both hits at ranks 1 and 2
    perfect_retrieved = [("DOC_A", 1), ("DOC_A", 2), ("DOC_B", 0)]
    assert math.isclose(compute_ndcg_at_k(perfect_retrieved, gold, k=2), 1.0)
    assert math.isclose(compute_ndcg_at_k(perfect_retrieved, gold, k=5), 1.0)

    # Inverted ranking: hit at rank 2, miss at rank 1 with 2 gold annotations
    # DCG@2 = 0 + 1.0 / log2(3). IDCG@2 with 2 gold items = 1.0 / log2(2) + 1.0 / log2(3).
    partial_retrieved = [("DOC_B", 0), ("DOC_A", 1), ("DOC_C", 5)]
    dcg = 1.0 / math.log2(3)
    idcg_dual = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    assert math.isclose(
        compute_ndcg_at_k(partial_retrieved, gold, k=2), dcg / idcg_dual, rel_tol=1e-5
    )

    # Single gold item case: IDCG@2 with 1 gold item = 1.0 / log2(2)
    single_gold = {("DOC_A", 1)}
    idcg_single = 1.0 / math.log2(2)
    assert math.isclose(
        compute_ndcg_at_k(partial_retrieved, single_gold, k=2),
        dcg / idcg_single,
        rel_tol=1e-5,
    )

    # Zero hits
    miss_retrieved = [("DOC_X", 0), ("DOC_Y", 1)]
    assert compute_ndcg_at_k(miss_retrieved, gold, k=5) == 0.0

    # Empty gold set
    assert compute_ndcg_at_k(perfect_retrieved, set(), k=5) == 0.0


def test_multiple_gold_evidence_annotations_evaluation():
    """Verify multi-evidence questions compute separate question coverage and annotation recall."""
    retriever_mock = MagicMock()
    # Mock retriever returns docA p5 at rank 1, docA p10 at rank 2
    retriever_mock.retrieve.return_value = [
        ChunkRetrievalResult("c0", "DOC_A", 5, 0, 1.0, 1, "text").to_dict(),
        ChunkRetrievalResult("c1", "DOC_A", 10, 0, 0.8, 2, "text").to_dict(),
    ]
    # Re-wrap as RetrievalResult objects
    from finveritas.rag_indexing.retrievers import RetrievalResult
    retriever_mock.retrieve.return_value = [
        RetrievalResult("DOC_A_p5", "DOC_A", 5, 1.0, 1, "text"),
        RetrievalResult("DOC_A_p10", "DOC_A", 10, 0.8, 2, "text"),
    ]

    question = [
        {
            "financebench_id": "FB_001",
            "question": "What were the revenues across both segments?",
            "doc_name": "DOC_A",
            "evidence": [
                {"evidence_doc_name": "DOC_A", "evidence_page_num": 5},
                {"evidence_doc_name": "DOC_A", "evidence_page_num": 18},  # Missed
            ],
        }
    ]

    eval_out = evaluate_retriever_on_financebench(
        retriever=retriever_mock,
        questions=question,
        indexed_doc_names={"DOC_A"},
        k_values=[1, 5],
    )
    total_perf = eval_out["performance"]["total_benchmark_metrics"]
    # Question is covered at cutoff 1 because p5 was retrieved
    assert total_perf["page_recall"]["recall_at_1"] == 1.0
    # But annotation recall at cutoff 1 is 1 / 2 = 0.5
    assert total_perf["annotation_recall"]["recall_at_1"] == 0.5


def test_paired_bootstrap_resampling_detects_difference():
    """Verify paired bootstrap resampling detects clear positive difference with narrow 95% CI."""
    # Baseline scores 0 on all 100 questions, Candidate scores 1 on all 100 questions
    base_evals = [
        QueryEvaluationResult(
            financebench_id=f"q{i}",
            question="q",
            gold_doc_name="d",
            gold_pages=[0],
            is_covered=True,
            uncovered_reason="",
            latency_ms=10.0,
            retrieved_results=[],
            doc_match_at_k={5: False},
            page_match_at_k={5: False},
            reciprocal_rank=0.0,
        )
        for i in range(100)
    ]
    cand_evals = [
        QueryEvaluationResult(
            financebench_id=f"q{i}",
            question="q",
            gold_doc_name="d",
            gold_pages=[0],
            is_covered=True,
            uncovered_reason="",
            latency_ms=10.0,
            retrieved_results=[],
            doc_match_at_k={5: True},
            page_match_at_k={5: True},
            reciprocal_rank=1.0,
        )
        for i in range(100)
    ]

    res = compute_paired_bootstrap(
        base_evals,
        cand_evals,
        metric_extractor=lambda r: 1.0 if r.page_match_at_k.get(5, False) else 0.0,
        n_bootstraps=500,
        seed=42,
    )
    assert res["mean_difference"] == 1.0
    assert res["ci_95"] == [1.0, 1.0]
    assert res["p_value"] == 0.0
    assert res["statistically_significant"] is True


def test_dense_retriever_cache_key_invalidation(tmp_path):
    """Verify DenseRetriever cache key alters when chunk size, count, or text changes."""
    retriever = DenseRetriever(model_name="BAAI/bge-small-en-v1.5", cache_dir=tmp_path)
    chunks_a = [
        ChunkCorpusItem("c0", "DOC", 0, 0, "text A", 0, 6, 2),
        ChunkCorpusItem("c1", "DOC", 0, 1, "text B", 6, 12, 2),
    ]
    chunks_b = [
        ChunkCorpusItem("c0", "DOC", 0, 0, "text A modified", 0, 15, 3),
        ChunkCorpusItem("c1", "DOC", 0, 1, "text B", 15, 21, 2),
    ]
    key_a = retriever._get_cache_key(chunks_a)
    key_b = retriever._get_cache_key(chunks_b)
    assert key_a != key_b


def test_numerical_question_classifier():
    """Verify numerical/table question classification logic."""
    assert is_numerical_question("What is the operating margin of 3M in 2022?") is True
    assert is_numerical_question("How much was capital expenditure in millions?") is True
    assert is_numerical_question("What is the debt to equity ratio for Adobe?") is True
    assert is_numerical_question("What is the general corporate strategy and mission?") is False
    assert is_numerical_question("Describe the business risk factors for cyber security.") is False


def test_missing_document_coverage_fallback():
    """Verify retriever evaluation cleanly marks missing documents as uncovered."""
    retriever_mock = MagicMock()
    questions = [
        {
            "financebench_id": "FB_999",
            "question": "What is the revenue?",
            "doc_name": "MISSING_DOC_2022_10K",
            "evidence": [{"evidence_doc_name": "MISSING_DOC_2022_10K", "evidence_page_num": 1}],
        }
    ]
    # Indexed docs does not contain MISSING_DOC_2022_10K
    eval_out = evaluate_retriever_on_financebench(
        retriever=retriever_mock,
        questions=questions,
        indexed_doc_names={"SOME_OTHER_DOC"},
        k_values=[1, 5],
    )
    cov = eval_out["benchmark_summary"]
    assert cov["covered_questions"] == 0
    assert cov["uncovered_questions"] == 1
    assert cov["coverage_fraction"] == 0.0


def test_dense_retriever_defaults_to_cpu():
    """Verify DenseRetriever strictly defaults to CPU execution after the system crash."""
    retriever = DenseRetriever()
    assert retriever.device == "cpu"
    assert retriever.batch_size == 32


def test_dense_retriever_incremental_cache_and_resume(tmp_path):
    """Verify incremental batch persistence, manifest writing, and resume without recomputation."""
    chunks = [
        ChunkCorpusItem(f"c{i}", "DOC_TEST", i // 2, i % 2, f"Financial sentence {i}", 0, 20, 3)
        for i in range(5)
    ]

    encode_calls = []

    def mock_encode(texts, batch_size=None, normalize_embeddings=True, show_progress_bar=False):
        encode_calls.append(len(texts))
        embs = np.zeros((len(texts), 384), dtype=np.float32)
        for idx, t in enumerate(texts):
            val = float(hash(t) % 1000) / 1000.0
            embs[idx, 0] = val
            embs[idx, 1] = 1.0
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    mock_model = MagicMock()
    mock_model.encode.side_effect = mock_encode

    retriever = DenseRetriever(
        batch_size=2,
        device="cpu",
        cache_dir=tmp_path,
        chunking_config={"chunk_size": 256, "chunk_overlap": 64},
    )
    retriever.model = mock_model

    retriever.fit(chunks)

    # 5 chunks with batch_size=2 => 3 batches (sizes 2, 2, 1)
    assert len(encode_calls) == 3
    assert encode_calls == [2, 2, 1]
    assert retriever.embeddings is not None
    assert retriever.embeddings.shape == (5, 384)

    # Verify unit norms
    norms = np.linalg.norm(retriever.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Verify cache folder contents
    cache_folder = retriever._get_cache_folder(chunks)
    assert cache_folder is not None
    assert (cache_folder / "manifest.json").is_file()
    assert (cache_folder / "embeddings.npy").is_file()
    assert (cache_folder / "batch_00000.npy").is_file()
    assert (cache_folder / "batch_00000.meta.json").is_file()
    assert (cache_folder / "batch_00001.npy").is_file()
    assert (cache_folder / "batch_00002.npy").is_file()

    # Second fit with a fresh retriever instance pointing to same cache
    retriever_resumed = DenseRetriever(
        batch_size=2,
        device="cpu",
        cache_dir=tmp_path,
    )
    retriever_resumed.model = MagicMock()
    retriever_resumed.model.encode.side_effect = lambda *args, **kwargs: pytest.fail(
        "Model encode should NOT be called on fully cached corpus"
    )

    retriever_resumed.fit(chunks)
    assert retriever_resumed.is_fitted is True
    assert retriever_resumed.embeddings is not None
    assert np.allclose(retriever_resumed.embeddings, retriever.embeddings)


def test_dense_retriever_corrupted_batch_detection_and_regeneration(tmp_path):
    """Verify corrupted/truncated batch files are detected, deleted, and safely regenerated."""
    chunks = [
        ChunkCorpusItem(f"c{i}", "DOC_TEST", 0, i, f"Financial sentence {i}", 0, 20, 3)
        for i in range(4)
    ]

    def mock_encode(texts, batch_size=None, normalize_embeddings=True, show_progress_bar=False):
        embs = np.ones((len(texts), 384), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    mock_model = MagicMock()
    mock_model.encode.side_effect = mock_encode

    retriever = DenseRetriever(batch_size=2, device="cpu", cache_dir=tmp_path)
    retriever.model = mock_model
    retriever.fit(chunks)

    cache_folder = retriever._get_cache_folder(chunks)
    assert cache_folder is not None

    # Corrupt batch 1 by truncating to 4 bytes and deleting embeddings.npy
    batch_1_file = cache_folder / "batch_00001.npy"
    with open(batch_1_file, "wb") as f:
        f.write(b"bad!")
    (cache_folder / "embeddings.npy").unlink()

    # Create new retriever; batch 0 should be reused, batch 1 regenerated
    retriever2 = DenseRetriever(batch_size=2, device="cpu", cache_dir=tmp_path)
    mock_model2 = MagicMock()
    calls = []

    def mock_encode2(texts, **kwargs):
        calls.append(len(texts))
        return mock_encode(texts)

    mock_model2.encode.side_effect = mock_encode2
    retriever2.model = mock_model2

    retriever2.fit(chunks)
    assert len(calls) == 1  # Only batch 1 was re-encoded!
    assert calls[0] == 2
    assert retriever2.embeddings is not None
    assert retriever2.embeddings.shape == (4, 384)


def test_dense_retriever_synthetic_query_retrieval(tmp_path):
    """Verify retrieval returns ranked, scored candidates with correct deduplication."""
    chunks = [
        ChunkCorpusItem("c0", "DOC_A", 1, 0, "Apple revenue grew by 15 percent", 0, 32, 7),
        ChunkCorpusItem("c1", "DOC_A", 1, 1, "Operating margin was 28 percent", 32, 64, 6),
        ChunkCorpusItem("c2", "DOC_B", 3, 0, "Microsoft cloud growth was high", 0, 32, 6),
    ]

    mock_model = MagicMock()

    def mock_encode(texts, normalize_embeddings=True, show_progress_bar=False, **kwargs):
        arr = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            if "Apple" in t:
                arr[i, 0] = 1.0
            elif "Operating" in t:
                arr[i, 1] = 1.0
            elif "Microsoft" in t:
                arr[i, 2] = 1.0
            else:
                arr[i, 0] = 0.8  # query vector favors Apple
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms

    mock_model.encode.side_effect = mock_encode
    retriever = DenseRetriever(batch_size=2, device="cpu", cache_dir=tmp_path)
    retriever.model = mock_model
    retriever.fit(chunks)

    # Retrieval of chunks
    c_results = retriever.retrieve_chunks("Apple financial results", top_k=2)
    assert len(c_results) == 2
    assert c_results[0].chunk_id == "c0"
    assert c_results[0].rank == 1

    # Retrieval of deduplicated pages
    p_results = retriever.retrieve("Apple financial results", top_k=2)
    assert len(p_results) >= 1
    assert p_results[0].doc_name == "DOC_A"
    assert p_results[0].page_index == 1
