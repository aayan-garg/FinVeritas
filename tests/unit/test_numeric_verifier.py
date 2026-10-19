"""
Unit tests for FinVeritas deterministic numerical financial claim verification.

Validates:
1. Scale normalization and equivalent representations (e.g. billion vs million).
2. Supported claims within explicit rounding boundaries.
3. Contradicted claims with matching metric and reporting period.
4. Reporting period mismatch handling.
5. Currency and denomination mismatch handling.
6. Multiple conflicting evidence candidates (fail closed).
7. Missing and unparseable evidence handling.
8. Rounding boundary edge cases and negative accounting values.
9. Citation metadata and PDF page numbering (0-based vs 1-based display).
10. CLI fallback when local corpus is unavailable.
11. Real filing verification on local AMD 2022 10-K if available.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from finveritas.verification import (
    EvidencePassage,
    FinancialClaim,
    VerificationVerdict,
    normalize_scale,
    verify_claim,
)
from scripts.verify_claim import resolve_worktree_data_path, run_claim_verification

# ---------------------------------------------------------------------------
# Test Case 1: Matching values using equivalent explicit scales
# ---------------------------------------------------------------------------


def test_scale_normalization_and_equivalent_representations() -> None:
    """Verifies that 23.6B, 23,600M, and 23,600,000K are mathematically identical."""
    assert normalize_scale("unit") == Decimal("1")
    assert normalize_scale("thousand") == Decimal("1000")
    assert normalize_scale("thousands") == Decimal("1000")
    assert normalize_scale("million") == Decimal("1000000")
    assert normalize_scale("millions") == Decimal("1000000")
    assert normalize_scale("billion") == Decimal("1000000000")
    assert normalize_scale("billions") == Decimal("1000000000")

    claim_billion = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
    )
    claim_million = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23600"),
        scale="million",
    )
    claim_thousand = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23600000"),
        scale="thousand",
    )

    expected = Decimal("23600000000")
    assert claim_billion.canonical_value == expected
    assert claim_million.canonical_value == expected
    assert claim_thousand.canonical_value == expected

    # Invalid scale raises ValueError
    with pytest.raises(ValueError, match="Unknown or unsupported scale"):
        normalize_scale("quadrillion_unsupported")


# ---------------------------------------------------------------------------
# Test Case 2: Supported claim with deterministic table extraction
# ---------------------------------------------------------------------------


def test_supported_claim_deterministic_verification() -> None:
    """Tests a supported claim against a multi-column financial statement table."""
    table_text = (
        "Advanced Micro Devices, Inc.\n"
        "Consolidated Statements of Operations\n"
        "(In millions, except per share amounts)\n"
        "Years Ended December 31, 2022 2021 2020\n"
        "Net revenue $ 23,601 $ 16,434 $ 9,763\n"
        "Cost of sales 11,550 8,505 5,416\n"
        "Operating income 1,264 3,648 1,369\n"
    )
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=table_text,
        source_url="https://sec.gov/amd_2022.pdf",
    )

    # Claim in billions: 23.6 billion (23,600M) vs filing 23,601M
    claim = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
        tolerance_pct=Decimal("0.005"),
    )

    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.matched_candidate is not None
    assert result.matched_candidate.canonical_value == Decimal("23601000000")
    assert len(result.citations) == 1
    assert result.citations[0].page_index == 53
    assert result.citations[0].display_page == 54
    assert "within documented tolerance" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 3: Contradicted claim with matching metric and period
# ---------------------------------------------------------------------------


def test_contradicted_claim_with_matching_metric_and_period() -> None:
    """Tests a contradicted claim where filing evidence unambiguously shows a conflict."""
    table_text = (
        "Advanced Micro Devices, Inc.\n"
        "Consolidated Statements of Operations\n"
        "(In millions, except per share amounts)\n"
        "Years Ended December 31, 2022 2021 2020\n"
        "Net revenue $ 23,601 $ 16,434 $ 9,763\n"
    )
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=table_text,
    )

    # False claim: 35.0 billion
    claim = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("35.0"),
        scale="billion",
        tolerance_pct=Decimal("0.005"),
    )

    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.CONTRADICTED
    assert result.matched_candidate is not None
    assert "materially contradicting" in result.explanation
    assert "+48.30%" in result.explanation or "+48.31%" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 4: Wrong reporting period
# ---------------------------------------------------------------------------


def test_wrong_reporting_period_insufficient_evidence() -> None:
    """Tests that a claim specifying a period not reported in the evidence fails closed."""
    table_text = (
        "Consolidated Statements of Operations\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021 2020\n"
        "Net revenue $ 23,601 $ 16,434 $ 9,763\n"
    )
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=table_text,
    )

    claim = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2024",  # Not in 2022 10-K
        value=Decimal("25.0"),
        scale="billion",
    )

    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "Reporting period mismatch" in result.explanation
    assert "2024" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 5: Currency / unit mismatch
# ---------------------------------------------------------------------------


def test_currency_and_unit_mismatch_insufficient_evidence() -> None:
    """Tests that a claim denominated in USD fails closed when evidence is denominated in EUR."""
    table_text = (
        "European Subsidiary Financial Statement\n"
        "(Amounts in millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Total revenue € 23,601 € 16,434\n"
    )
    passage = EvidencePassage(
        source_doc="EURO_CORP_2022_10K",
        page_index=12,
        text=table_text,
    )

    claim = FinancialClaim(
        entity="EURO_CORP",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
        currency="USD",  # Claim specifies USD but filing is EUR
    )

    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "Currency mismatch" in result.explanation
    assert "USD" in result.explanation
    assert "EUR" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 6: Multiple conflicting evidence candidates
# ---------------------------------------------------------------------------


def test_multiple_conflicting_evidence_candidates() -> None:
    """Tests that ambiguous evidence reporting conflicting figures fails closed."""
    passage_text = (
        "Segment Disclosures\n"
        "Revenue for 2022 was $23.6 billion across all primary products.\n"
        "In our adjusted summary, revenue for 2022 was $45.0 billion after pro-forma mergers.\n"
    )
    passage = EvidencePassage(
        source_doc="CONF_CORP_2022_10K",
        page_index=30,
        text=passage_text,
    )

    claim = FinancialClaim(
        entity="CONF_CORP",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
    )

    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "Multiple conflicting evidence figures" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 7: Missing and unparseable evidence
# ---------------------------------------------------------------------------


def test_missing_and_unparseable_evidence() -> None:
    """Tests behavior on missing passages, unparseable text, or unsupported metrics."""
    claim = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
    )

    # Empty evidence list
    res_empty = verify_claim(claim, [])
    assert res_empty.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "No evidence passages provided" in res_empty.explanation

    # Unparseable narrative without financial figures
    unparseable_passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=1,
        text="The compensation committee met quarterly to review human capital development.",
    )
    res_unparseable = verify_claim(claim, [unparseable_passage])
    assert res_unparseable.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "No structured financial values" in res_unparseable.explanation

    # Unsupported metric
    unsupported_claim = FinancialClaim(
        entity="AMD",
        metric="net_promoter_score",
        reporting_period="2022",
        value=Decimal("75"),
    )
    res_unsupported = verify_claim(unsupported_claim, [unparseable_passage])
    assert res_unsupported.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "Unsupported metric" in res_unsupported.explanation


# ---------------------------------------------------------------------------
# Test Case 8: Rounding boundaries and negative accounting values
# ---------------------------------------------------------------------------


def test_rounding_boundaries_and_negative_accounting_values() -> None:
    """Tests half-step rounding boundary logic and accounting parentheses for negative figures."""
    # Rounding boundary check: stated 100.0 (half-step 0.05 -> [99.95, 100.05])
    claim_boundary = FinancialClaim(
        entity="ACME",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("100.0"),
        scale="million",
        tolerance_pct=Decimal("0.001"),  # 0.1% tolerance
    )
    lower_bound, upper_bound = claim_boundary.rounding_bounds
    assert lower_bound == Decimal("99950000.0")
    assert upper_bound == Decimal("100050000.0")

    # Negative accounting value in parentheses: (5,053) million = -$5.053B
    boeing_table = (
        "The Boeing Company and Subsidiaries\n"
        "Consolidated Statements of Operations\n"
        "(Dollars in millions, except per share data)\n"
        "Years ended December 31, 2022 2021 2020\n"
        "Net loss (5,053) (4,290) (11,941)\n"
    )
    passage = EvidencePassage(
        source_doc="BOEING_2022_10K",
        page_index=54,
        text=boeing_table,
    )

    loss_claim = FinancialClaim(
        entity="Boeing",
        metric="net income",
        reporting_period="2022",
        value=Decimal("-5053"),
        scale="million",
    )

    res_loss = verify_claim(loss_claim, [passage])
    assert res_loss.verdict == VerificationVerdict.SUPPORTED
    assert res_loss.matched_candidate is not None
    assert res_loss.matched_candidate.nominal_value == Decimal("-5053")
    assert res_loss.matched_candidate.canonical_value == Decimal("-5053000000")
    assert res_loss.matched_candidate.is_negative is True


# ---------------------------------------------------------------------------
# Test Case 9: Citation metadata and PDF page numbering
# ---------------------------------------------------------------------------


def test_citation_metadata_and_pdf_page_numbering() -> None:
    """Verifies that citations preserve 0-based page_index and 1-based display_page."""
    text = (
        "Consolidated Statements of Operations\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Operating income $ 1,264 $ 3,648\n"
    )
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=text,
        source_url="https://ir.amd.com/sec-filings/amd-2022-10k.pdf",
    )

    claim = FinancialClaim(
        entity="AMD",
        metric="operating income",
        reporting_period="2022",
        value=Decimal("1264"),
        scale="million",
    )

    res = verify_claim(claim, [passage])
    assert res.verdict == VerificationVerdict.SUPPORTED
    assert len(res.citations) == 1

    cite = res.citations[0]
    assert cite.source_doc == "AMD_2022_10K"
    assert cite.page_index == 53
    assert cite.display_page == 54  # Strictly page_index + 1
    assert cite.source_url == "https://ir.amd.com/sec-filings/amd-2022-10k.pdf"
    assert "Operating income" in cite.matched_text


# ---------------------------------------------------------------------------
# Test Case 10: CLI fallback when local corpus is unavailable
# ---------------------------------------------------------------------------


def test_cli_fallback_when_local_corpus_unavailable(tmp_path: Path) -> None:
    """Verifies that the CLI fails gracefully if filing corpus is missing."""
    empty_pdf_dir = tmp_path / "empty_pdfs"
    empty_pdf_dir.mkdir(parents=True)

    result = run_claim_verification(
        entity="NONEXISTENT_CORP",
        metric="revenue",
        period="2022",
        value="100.0",
        scale="million",
        pdf_dir=empty_pdf_dir,
        metadata_path=tmp_path / "missing.jsonl",
    )

    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "not found" in result.explanation
    assert "Local filing corpus unavailable" in result.limitations[0]


# ---------------------------------------------------------------------------
# Test Case 11: Real filing verification on AMD 2022 10-K
# ---------------------------------------------------------------------------


def test_real_filing_verified_example_amd_2022() -> None:
    """Tests end-to-end retrieval and verification on actual local AMD 2022 10-K filing."""
    pdf_dir = resolve_worktree_data_path(Path("data/raw/financebench/pdfs"))
    amd_pdf = pdf_dir / "AMD_2022_10K.pdf"

    if not amd_pdf.is_file():
        pytest.skip(f"Local AMD filing not found at {amd_pdf}")

    result = run_claim_verification(
        entity="AMD",
        doc_name="AMD_2022_10K",
        metric="revenue",
        period="2022",
        value="23.6",
        scale="billion",
        pdf_dir=pdf_dir,
    )

    assert result.verdict == VerificationVerdict.SUPPORTED
    assert len(result.citations) >= 1
    assert result.citations[0].source_doc == "AMD_2022_10K"
    assert result.citations[0].page_index in (42, 53)
    assert result.matched_candidate is not None
    assert result.matched_candidate.canonical_value in (
        Decimal("23600000000"),
        Decimal("23601000000"),
    )
