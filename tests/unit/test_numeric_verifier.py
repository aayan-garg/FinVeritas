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

from finveritas.rag_indexing.financebench_data import extract_pdf_pages
from finveritas.verification import (
    MAX_PERMITTED_TOLERANCE,
    EvidencePassage,
    FinancialClaim,
    VerificationVerdict,
    normalize_scale,
    validate_entity_match,
    verify_claim,
)
from scripts.verify_claim import (
    check_claim_text_inconsistencies,
    format_cli_output,
    resolve_worktree_data_path,
    run_claim_verification,
)

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
        "Consolidated Statements of Operations\n"
        "Total revenue for 2022 was $23.6 billion across all reporting units.\n"
        "In note 15, consolidated revenue for 2022 was reported as $45.0 billion.\n"
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


# ---------------------------------------------------------------------------
# Test Case 12: Entity validation — Boeing claim against AMD filing fails closed
# ---------------------------------------------------------------------------


def test_entity_validation_boeing_claim_against_amd_filing_fails_closed() -> None:
    """Verifies that specifying an AMD filing for a Boeing claim fails closed."""
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=(
            "Consolidated Statements of Operations\n"
            "Years Ended December 31, 2022 2021 2020\n"
            "(In millions)\n"
            "Net revenue $ 23,601 $ 16,434 $ 9,763\n"
        ),
        metadata={"company": "Advanced Micro Devices, Inc."},
    )
    claim = FinancialClaim(
        entity="Boeing",
        doc_name="AMD_2022_10K",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
    )
    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "Entity validation failed" in result.explanation
    assert "Boeing" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 13: Entity validation — AMD claim against AMD filing proceeds
# ---------------------------------------------------------------------------


def test_entity_validation_amd_claim_against_amd_filing_proceeds() -> None:
    """Verifies that an AMD claim correctly validates against an AMD filing."""
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=(
            "Consolidated Statements of Operations\n"
            "Years Ended December 31, 2022 2021 2020\n"
            "(In millions)\n"
            "Net revenue $ 23,601 $ 16,434 $ 9,763\n"
        ),
        metadata={"company": "Advanced Micro Devices, Inc."},
    )
    claim = FinancialClaim(
        entity="Advanced Micro Devices",
        doc_name="AMD_2022_10K",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
    )
    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.matched_candidate is not None
    assert result.matched_candidate.canonical_value == Decimal("23601000000")


# ---------------------------------------------------------------------------
# Test Case 14: Entity validation — Unsupported or ambiguous mapping fails closed
# ---------------------------------------------------------------------------


def test_entity_validation_unsupported_or_ambiguous_mapping_fails_closed() -> None:
    """Verifies that vague or ambiguous entity-to-document mappings fail closed."""
    # Vague substring attempt: 'BA' against 'ALIBABA'
    valid, reason = validate_entity_match("BA", "ALIBABA_2022_10K")
    assert valid is False
    assert "Entity mismatch" in reason

    # Unknown entity fails closed
    passage = EvidencePassage(
        source_doc="UNKNOWN_DOC_2022_10K",
        page_index=1,
        text=(
            "Consolidated Statements of Operations\n"
            "Years Ended December 31, 2022 2021\n"
            "(In millions)\n"
            "Net revenue $ 100 $ 90\n"
        ),
    )
    claim = FinancialClaim(
        entity="RandomUnregisteredCo",
        doc_name="UNKNOWN_DOC_2022_10K",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("100"),
        scale="million",
    )
    res = verify_claim(claim, [passage])
    assert res.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "Entity validation failed" in res.explanation


# ---------------------------------------------------------------------------
# Test Case 15: Entity validation — Conflicting metadata rejected
# ---------------------------------------------------------------------------


def test_entity_validation_conflicting_metadata_rejected() -> None:
    """Verifies that passages with metadata explicitly attributing to another firm are rejected."""
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=(
            "Consolidated Statements of Operations\n"
            "Years Ended December 31, 2022 2021\n"
            "(In millions)\n"
            "Net revenue $ 23,601 $ 16,434\n"
        ),
        metadata={"company": "The Boeing Company"},  # Deliberate conflict
    )
    claim = FinancialClaim(
        entity="AMD",
        doc_name="AMD_2022_10K",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
    )
    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert "Conflicting entity metadata" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 16: Zero and numeric edge — Zero claim and zero evidence supported
# ---------------------------------------------------------------------------


def test_zero_claim_and_zero_evidence_value_supported() -> None:
    """Verifies that a claim of zero against reported zero evidence is supported."""
    table_text = (
        "Consolidated Statements of Operations\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Operating income $ 0 $ 500\n"
    )
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=table_text,
    )
    claim = FinancialClaim(
        entity="AMD",
        metric="operating income",
        reporting_period="2022",
        value=Decimal("0"),
        scale="million",
    )
    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.matched_candidate is not None
    assert result.matched_candidate.canonical_value == Decimal("0")
    assert "exact zero match" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 17: Zero and numeric edge — Nonzero claim vs zero evidence contradicts without crash
# ---------------------------------------------------------------------------


def test_nonzero_claim_against_zero_evidence_contradicted_no_crash() -> None:
    """
    Verifies that a nonzero claim against zero evidence never crashes and returns CONTRADICTED.
    """
    table_text = (
        "Consolidated Statements of Operations\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Operating income $ 0 $ 500\n"
    )
    passage = EvidencePassage(
        source_doc="AMD_2022_10K",
        page_index=53,
        text=table_text,
    )
    claim = FinancialClaim(
        entity="AMD",
        metric="operating income",
        reporting_period="2022",
        value=Decimal("1500"),
        scale="million",
    )
    # Must never raise ZeroDivisionError
    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.CONTRADICTED
    assert "against zero evidence" in result.explanation
    assert "relative difference undefined" in result.explanation


# ---------------------------------------------------------------------------
# Test Case 18: Tolerance parameter validation
# ---------------------------------------------------------------------------


def test_tolerance_parameter_validation() -> None:
    """Verifies that invalid or negative tolerances are rejected with clear ValueErrors."""
    # Negative tolerance rejected
    with pytest.raises(ValueError, match="tolerance_pct cannot be negative"):
        FinancialClaim(
            entity="AMD",
            metric="revenue",
            reporting_period="2022",
            value=Decimal("23.6"),
            scale="billion",
            tolerance_pct=Decimal("-0.01"),
        )

    # Exceeds MAX_PERMITTED_TOLERANCE (0.20 = 20%)
    with pytest.raises(ValueError, match="exceeds maximum permitted tolerance"):
        FinancialClaim(
            entity="AMD",
            metric="revenue",
            reporting_period="2022",
            value=Decimal("23.6"),
            scale="billion",
            tolerance_pct=Decimal("0.25"),
        )

    # Valid tolerance <= 0.20 accepted
    valid_claim = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
        tolerance_pct=Decimal("0.10"),
    )
    assert valid_claim.tolerance_pct == Decimal("0.10")
    assert MAX_PERMITTED_TOLERANCE == Decimal("0.20")


# ---------------------------------------------------------------------------
# Test Case 19: Negative accounting values and rounding boundaries
# ---------------------------------------------------------------------------


def test_negative_values_and_rounding_boundaries() -> None:
    """Tests negative accounting values and exact Decimal interval boundaries."""
    # Negative claim with 1 decimal place: -5.0 billion
    # Half step is 0.05 billion -> [-5.05e9, -4.95e9]
    neg_claim = FinancialClaim(
        entity="Boeing",
        metric="net income",
        reporting_period="2022",
        value=Decimal("-5.0"),
        scale="billion",
        tolerance_pct=Decimal("0.005"),
    )
    low, high = neg_claim.rounding_bounds
    assert low == Decimal("-5050000000.0")
    assert high == Decimal("-4950000000.0")

    table_text = (
        "Consolidated Statements of Operations\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Net loss (5,053) (4,290)\n"
    )
    passage = EvidencePassage(
        source_doc="BOEING_2022_10K",
        page_index=54,
        text=table_text,
    )
    res = verify_claim(neg_claim, [passage])
    # -5053M is outside [-5050M, -4950M], and relative difference is 1.05% > 0.5% tolerance
    assert res.verdict == VerificationVerdict.CONTRADICTED


# ---------------------------------------------------------------------------
# Test Case 20: Adversarial fixtures — Metric scope disambiguation
# ---------------------------------------------------------------------------


def test_adversarial_metric_scope_disambiguation_fixtures() -> None:
    """Verifies that non-consolidated rows are disqualified for core consolidated claims."""
    # 1. Segment revenue must NOT match consolidated revenue claim
    seg_text = (
        "Consolidated Statements of Operations and Segment Information\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Client segment revenue $ 6,043 $ 4,200\n"
    )
    pass_seg = EvidencePassage(source_doc="AMD_2022_10K", page_index=80, text=seg_text)
    claim_rev = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("6.043"),
        scale="billion",
    )
    res_seg = verify_claim(claim_rev, [pass_seg])
    assert res_seg.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE

    # 2. Earnings per share must NOT match net income claim
    eps_text = (
        "Consolidated Statements of Operations\n"
        "Years Ended December 31, 2022 2021\n"
        "Earnings per share - Basic $ 0.85 $ 2.61\n"
        "Earnings per share - Diluted $ 0.84 $ 2.57\n"
    )
    pass_eps = EvidencePassage(source_doc="AMD_2022_10K", page_index=53, text=eps_text)
    claim_ni = FinancialClaim(
        entity="AMD",
        metric="net income",
        reporting_period="2022",
        value=Decimal("0.85"),
        scale="unit",
    )
    res_eps = verify_claim(claim_ni, [pass_eps])
    assert res_eps.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE

    # 3. Adjusted operating income must NOT match operating income claim
    adj_text = (
        "Non-GAAP Reconciliations\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Adjusted operating income $ 1,500 $ 3,000\n"
    )
    pass_adj = EvidencePassage(source_doc="AMD_2022_10K", page_index=45, text=adj_text)
    claim_oi = FinancialClaim(
        entity="AMD",
        metric="operating income",
        reporting_period="2022",
        value=Decimal("1.5"),
        scale="billion",
    )
    res_adj = verify_claim(claim_oi, [pass_adj])
    assert res_adj.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE

    # 4. Continuing operations must NOT match net income claim
    cont_text = (
        "Consolidated Statements of Operations\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Income from continuing operations $ 1,184 $ 3,669\n"
    )
    pass_cont = EvidencePassage(source_doc="AMD_2022_10K", page_index=53, text=cont_text)
    claim_cont = FinancialClaim(
        entity="AMD",
        metric="net income",
        reporting_period="2022",
        value=Decimal("1184"),
        scale="million",
    )
    res_cont = verify_claim(claim_cont, [pass_cont])
    assert res_cont.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE

    # 5. Cost of sales must NOT match revenue claim
    cost_text = (
        "Consolidated Statements of Operations\n"
        "(In millions)\n"
        "Years Ended December 31, 2022 2021\n"
        "Cost of sales $ 11,550 $ 8,505\n"
    )
    pass_cost = EvidencePassage(source_doc="AMD_2022_10K", page_index=53, text=cost_text)
    claim_cost = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("11550"),
        scale="million",
    )
    res_cost = verify_claim(claim_cost, [pass_cost])
    assert res_cost.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# Test Case 21: Real filing statement-table scale conversion on AMD 2022 10-K
# ---------------------------------------------------------------------------


def test_real_filing_statement_table_scale_conversion_amd_2022() -> None:
    """
    Supplies the exact statement-table passage from AMD 2022 10-K reporting $23,601 million
    for 2022 and verifies the equivalent claim of $23.6 billion.
    Asserts table row source type, canonical value 23,601,000,000 USD, page index 53, and
    verifies that narrative passages are not used.
    """
    pdf_dir = resolve_worktree_data_path(Path("data/raw/financebench/pdfs"))
    amd_pdf = pdf_dir / "AMD_2022_10K.pdf"
    if not amd_pdf.is_file():
        pytest.skip(f"Local AMD filing not found at {amd_pdf}")

    pages = extract_pdf_pages(amd_pdf, "AMD_2022_10K")
    p53_list = [p for p in pages if p.page_index == 53]
    assert len(p53_list) == 1, "Page 53 must be present in extracted AMD 2022 10-K"
    p53 = p53_list[0]

    passage = EvidencePassage(
        source_doc=p53.doc_name,
        page_index=p53.page_index,
        text=p53.text,
        source_url="https://sec.gov/edgar/amd-2022-10k.htm",
        metadata={"company": "Advanced Micro Devices, Inc."},
    )

    claim = FinancialClaim(
        entity="AMD",
        doc_name="AMD_2022_10K",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
        tolerance_pct=Decimal("0.005"),
    )

    result = verify_claim(claim, [passage])
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.matched_candidate is not None

    # Strict Checkpoint 10 assertions:
    # 1. Extracted candidate is table row, not narrative
    assert result.matched_candidate.source_type == "table_row"
    assert result.matched_candidate.source_type != "narrative"

    # 2. Selected reporting period is 2022
    assert result.matched_candidate.reporting_period == "2022"

    # 3. Scale is million, nominal value is 23,601, canonical value is 23,601,000,000 USD
    assert result.matched_candidate.scale == "million"
    assert result.matched_candidate.nominal_value == Decimal("23601")
    assert result.matched_candidate.canonical_value == Decimal("23601000000")

    # 4. Supports the rounded claim under precision rule
    assert "within documented tolerance" in result.explanation

    # 5. Citation matches precise source page and includes the relevant row
    assert len(result.citations) == 1
    assert result.citations[0].page_index == 53
    assert result.citations[0].display_page == 54
    assert "Net revenue" in result.citations[0].matched_text


# ---------------------------------------------------------------------------
# Test Case 22: Claim-input semantics and inconsistency detection
# ---------------------------------------------------------------------------


def test_claim_input_semantics_and_inconsistencies() -> None:
    """Verifies that non-authoritative claim text discrepancies are detected and labeled."""
    # Period discrepancy: claim text says 2021 but structured period is 2022
    warn_period = check_claim_text_inconsistencies(
        claim_text="AMD 2021 annual revenue was 23.6B",
        entity="AMD",
        metric="revenue",
        period="2022",
        value="23.6",
    )
    assert len(warn_period) >= 1
    assert "Period discrepancy" in warn_period[0]

    # Entity discrepancy: claim text mentions Boeing but entity is AMD
    warn_entity = check_claim_text_inconsistencies(
        claim_text="Boeing reported revenue of 23.6 billion",
        entity="AMD",
        metric="revenue",
        period="2022",
        value="23.6",
    )
    assert len(warn_entity) >= 1
    assert "Entity discrepancy" in warn_entity[0]

    # Consistent claim text
    warn_none = check_claim_text_inconsistencies(
        claim_text="AMD reported 2022 net revenue of 23.6 billion",
        entity="AMD",
        metric="revenue",
        period="2022",
        value="23.6",
    )
    assert len(warn_none) == 0

    # Verify format_cli_output clearly labels descriptive text
    claim = FinancialClaim(
        entity="AMD",
        metric="revenue",
        reporting_period="2022",
        value=Decimal("23.6"),
        scale="billion",
    )
    dummy_res = verify_claim(claim, [])
    out = format_cli_output(dummy_res, claim_text="Descriptive summary text")
    assert "Non-Authoritative Text" in out
    assert "descriptive only" in out
