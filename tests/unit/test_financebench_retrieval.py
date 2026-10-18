"""
FinVeritas: Unit Tests for FinanceBench Evidence Retrieval.

Tests document/page identity, zero-based page indexing conventions,
conservative text normalization (preserving financial notation),
TF-IDF and BM25 ranking mechanics, evaluation metric calculations,
and missing-source (uncovered question) accounting.
"""

import unittest

from finveritas.rag_indexing import (
    BM25Retriever,
    PageCorpusItem,
    TfidfRetriever,
    evaluate_retriever_on_financebench,
    normalize_financial_text,
    tokenize_financial_query,
)


class TestFinanceBenchRetrieval(unittest.TestCase):
    def setUp(self):
        # Create a controlled synthetic page corpus across 3 companies
        self.corpus = [
            PageCorpusItem(
                page_id="CORP_A_2022_10K_p0",
                doc_name="CORP_A_2022_10K",
                page_index=0,
                text="Cover page for Corp A annual report for fiscal year 2022.",
                doc_link="https://sec.gov/corp_a_2022",
                char_count=57,
            ),
            PageCorpusItem(
                page_id="CORP_A_2022_10K_p48",
                doc_name="CORP_A_2022_10K",
                page_index=48,
                text=(
                    "Consolidated Balance Sheet. Total assets were $46,455 million in 2022 "
                    "compared to $47,072 million in 2021. Cash and cash equivalents: $3,655M."
                ),
                doc_link="https://sec.gov/corp_a_2022",
                char_count=165,
            ),
            PageCorpusItem(
                page_id="CORP_A_2022_10K_p50",
                doc_name="CORP_A_2022_10K",
                page_index=50,
                text=(
                    "Consolidated Statement of Cash Flows. "
                    "Capital expenditures were $(1,749) million. "
                    "Operating cash flow was $5,591 million. Dividends paid were $3,369M."
                ),
                doc_link="https://sec.gov/corp_a_2022",
                char_count=182,
            ),
            PageCorpusItem(
                page_id="CORP_B_2021_10K_p12",
                doc_name="CORP_B_2021_10K",
                page_index=12,
                text=(
                    "Risk Factors. We face severe cybersecurity risks and infrastructure flaws. "
                    "Operating margin compressed by 3.5% due to supply chain inflation."
                ),
                doc_link="https://sec.gov/corp_b_2021",
                char_count=189,
            ),
            PageCorpusItem(
                page_id="CORP_C_2023Q2_10Q_p5",
                doc_name="CORP_C_2023Q2_10Q",
                page_index=5,
                text=(
                    "Segment Revenue. North America enterprise cloud revenue grew 14.2% to $12.8B. "
                    "Operating income for the quarter reached $3.2B."
                ),
                doc_link="https://sec.gov/corp_c_2023q2",
                char_count=144,
            ),
        ]

    def test_page_corpus_item_identity_and_indexing(self):
        item = self.corpus[1]
        self.assertEqual(item.doc_name, "CORP_A_2022_10K")
        self.assertEqual(item.page_index, 48)  # 0-indexed
        self.assertEqual(item.page_id, "CORP_A_2022_10K_p48")
        self.assertIn("$46,455 million", item.text)

    def test_conservative_financial_normalization(self):
        dirty_text = (
            "Consolidated \r\n Balance   Sheet\u00a0At December 31\n\n\n\n"
            "Total Assets:  $46,455 \t\t Capital Exp: $(1,749) million  Margin: -2.4%  €50M"
        )
        cleaned = normalize_financial_text(dirty_text)

        # Check preserved financial notations
        self.assertIn("$46,455", cleaned)
        self.assertIn("$(1,749)", cleaned)
        self.assertIn("-2.4%", cleaned)
        self.assertIn("€50M", cleaned)

        # Check whitespace cleanup
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\u00a0", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Consolidated\nBalance Sheet At December 31", cleaned)

    def test_tokenize_financial_query(self):
        tokens = tokenize_financial_query("What was Corp-A's Q3 capex ($ millions) and margin %?")
        self.assertIn("corp-a", tokens)
        self.assertIn("q3", tokens)
        self.assertIn("capex", tokens)
        self.assertIn("$", tokens)
        self.assertIn("%", tokens)

    def test_tfidf_retriever_ranking(self):
        retriever = TfidfRetriever(min_df=1)
        retriever.fit(self.corpus)

        results = retriever.retrieve(
            "What was Corp A total assets on balance sheet in 2022?",
            top_k=3,
        )
        self.assertEqual(len(results), 3)

        top_hit = results[0]
        self.assertEqual(top_hit.doc_name, "CORP_A_2022_10K")
        self.assertEqual(top_hit.page_index, 48)
        self.assertGreater(top_hit.score, 0.0)
        self.assertEqual(top_hit.rank, 1)

    def test_bm25_retriever_ranking(self):
        retriever = BM25Retriever()
        retriever.fit(self.corpus)

        results = retriever.retrieve(
            "What were capital expenditures and cash flows in 2022?",
            top_k=3,
        )
        self.assertEqual(len(results), 3)

        top_hit = results[0]
        self.assertEqual(top_hit.doc_name, "CORP_A_2022_10K")
        self.assertEqual(top_hit.page_index, 50)
        self.assertGreater(top_hit.score, 0.0)

    def test_gold_page_and_document_matching_metrics(self):
        retriever = BM25Retriever()
        retriever.fit(self.corpus)
        indexed_docs = set(item.doc_name for item in self.corpus)

        questions = [
            {
                "financebench_id": "financebench_001",
                "question": "What were the capital expenditures in 2022 for Corp A?",
                "doc_name": "CORP_A_2022_10K",
                "evidence": [{"doc_name": "CORP_A_2022_10K", "evidence_page_num": 50}],
            },
            {
                "financebench_id": "financebench_002",
                "question": "What cybersecurity risk factors are reported by Corp B?",
                "doc_name": "CORP_B_2021_10K",
                "evidence": [{"doc_name": "CORP_B_2021_10K", "evidence_page_num": 12}],
            },
        ]

        eval_res = evaluate_retriever_on_financebench(
            retriever=retriever,
            questions=questions,
            indexed_doc_names=indexed_docs,
            k_values=[1, 3, 5],
        )

        perf = eval_res["performance"]["eligible_subset_metrics"]
        self.assertEqual(perf["denominator"], 2)
        # Both questions hit gold document at top 1
        self.assertEqual(perf["doc_recall"]["recall_at_1"], 1.0)
        # Q2 hits at rank 1, Q1 hits at rank 2 (page recall@1 is 0.5, recall@3 is 1.0)
        self.assertEqual(perf["page_recall"]["recall_at_1"], 0.5)
        self.assertEqual(perf["page_recall"]["recall_at_3"], 1.0)
        self.assertEqual(perf["page_recall"]["recall_at_5"], 1.0)
        # MRR = (1/2 + 1/1) / 2 = 0.75
        self.assertEqual(perf["mrr"], 0.75)

    def test_uncovered_question_handling(self):
        retriever = BM25Retriever()
        retriever.fit(self.corpus)
        indexed_docs = set(item.doc_name for item in self.corpus)

        # Question 1 is covered, Question 2 refers to a missing PDF
        questions = [
            {
                "financebench_id": "financebench_covered",
                "question": "What was North America enterprise cloud revenue for Corp C?",
                "doc_name": "CORP_C_2023Q2_10Q",
                "evidence": [{"doc_name": "CORP_C_2023Q2_10Q", "evidence_page_num": 5}],
            },
            {
                "financebench_id": "financebench_uncovered",
                "question": "What was the operating margin of Corp Z in 2019?",
                "doc_name": "CORP_Z_2019_10K",  # Not in indexed_docs
                "evidence": [{"doc_name": "CORP_Z_2019_10K", "evidence_page_num": 25}],
            },
        ]

        eval_res = evaluate_retriever_on_financebench(
            retriever=retriever,
            questions=questions,
            indexed_doc_names=indexed_docs,
            k_values=[1, 3, 5],
        )

        summary = eval_res["benchmark_summary"]
        self.assertEqual(summary["total_benchmark_questions"], 2)
        self.assertEqual(summary["covered_questions"], 1)
        self.assertEqual(summary["uncovered_questions"], 1)
        self.assertEqual(summary["coverage_fraction"], 0.5)

        # Eligible subset should have 1 denominator with 1.0 recall
        elig = eval_res["performance"]["eligible_subset_metrics"]
        self.assertEqual(elig["denominator"], 1)
        self.assertEqual(elig["page_recall"]["recall_at_1"], 1.0)

        # Total benchmark should have 2 denominator with 0.5 recall (penalizing uncovered question)
        tot = eval_res["performance"]["total_benchmark_metrics"]
        self.assertEqual(tot["denominator"], 2)
        self.assertEqual(tot["page_recall"]["recall_at_1"], 0.5)
        self.assertEqual(tot["mrr"], 0.5)


if __name__ == "__main__":
    unittest.main()
