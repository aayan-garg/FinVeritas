#!/usr/bin/env python3
"""
FinVeritas: CLI for Evidence-Grounded Financial Claim Verification.

Connects local financial PDF corpus retrieval (via BM25) with the deterministic
numerical claim verifier to validate quantitative claims against SEC filings.

Usage examples:
    # Supported claim on AMD 2022 Revenue:
    python scripts/verify_claim.py --entity AMD --doc AMD_2022_10K \
        --metric "revenue" --period 2022 --value 23.6 --scale billion

    # Equivalent scaled claim:
    python scripts/verify_claim.py --entity AMD --doc AMD_2022_10K \
        --metric "revenue" --period 2022 --value 23601 --scale million

    # Contradicted claim on Boeing 2022 Revenue:
    python scripts/verify_claim.py --entity Boeing --doc BOEING_2022_10K \
        --metric "revenue" --period 2022 --value 95.0 --scale billion
"""


from __future__ import annotations

import argparse
import json
import logging
import sys
from decimal import Decimal
from pathlib import Path

from finveritas.rag_indexing.financebench_data import (
    PageCorpusItem,
    extract_pdf_pages,
)
from finveritas.rag_indexing.retrievers import BM25Retriever
from finveritas.verification import (
    EvidencePassage,
    FinancialClaim,
    VerificationResult,
    VerificationVerdict,
    verify_claim,
)

logger = logging.getLogger("finveritas.cli.verify_claim")


def resolve_worktree_data_path(rel_path: Path) -> Path:
    """Resolves data paths across main repo and git worktrees."""
    if rel_path.exists():
        return rel_path
    script_dir = Path(__file__).resolve().parent
    worktree_root = script_dir.parent
    if ".worktrees" in str(worktree_root):
        # Worktree is located under <repo>/.worktrees/<name>
        main_root = worktree_root.parent.parent
        main_path = main_root / rel_path
        if main_path.exists():
            return main_path
    return rel_path


def resolve_filing_pdf(
    entity: str,
    period: str,
    doc_name: str | None,
    pdf_dir: Path,
) -> tuple[Path | None, str]:
    """
    Locates the corresponding PDF filing in the local directory.
    If doc_name is explicitly provided, checks for {doc_name}.pdf.
    Otherwise searches for matching filings by entity and period.
    """
    pdf_dir = resolve_worktree_data_path(pdf_dir)
    if not pdf_dir.is_dir():
        return None, doc_name or f"{entity}_{period}"

    if doc_name:
        direct_path = pdf_dir / f"{doc_name}.pdf"
        if direct_path.is_file():
            return direct_path, doc_name
        # Case insensitive match
        for f in pdf_dir.glob("*.pdf"):
            if f.stem.lower() == doc_name.lower():
                return f, f.stem

    # Search by entity and year
    ent_norm = entity.replace(" ", "").lower()
    for f in pdf_dir.glob("*.pdf"):
        stem_lower = f.stem.lower()
        if ent_norm in stem_lower and period in stem_lower:
            return f, f.stem

    return None, doc_name or f"{entity}_{period}"


def load_doc_link_map(metadata_path: Path | None) -> dict[str, str]:
    """Loads document information map if local metadata JSONL is present."""
    if not metadata_path:
        return {}
    metadata_path = resolve_worktree_data_path(metadata_path)
    if not metadata_path.is_file():
        return {}
    doc_links: dict[str, str] = {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                info = json.loads(line_str)
                name = info.get("doc_name")
                link = info.get("doc_link")
                if name and link:
                    doc_links[name] = link
    except Exception as err:
        logger.debug(f"Could not load doc links: {err}")
    return doc_links


def run_claim_verification(
    entity: str,
    metric: str,
    period: str,
    value: str,
    scale: str = "unit",
    currency: str = "USD",
    doc_name: str | None = None,
    tolerance: str = "0.005",
    pdf_dir: Path = Path("data/raw/financebench/pdfs"),
    metadata_path: Path | None = Path(
        "data/raw/financebench/financebench_document_information.jsonl"
    ),
    top_k: int = 5,
) -> VerificationResult:
    """
    Coordinates local retrieval and deterministic claim verification.
    """
    claim = FinancialClaim(
        entity=entity,
        metric=metric,
        reporting_period=period,
        value=Decimal(value),
        scale=scale,
        currency=currency,
        doc_name=doc_name,
        tolerance_pct=Decimal(tolerance),
    )

    # Check local filing PDF availability
    pdf_path, resolved_doc = resolve_filing_pdf(
        entity=entity,
        period=period,
        doc_name=doc_name,
        pdf_dir=pdf_dir,
    )

    if not pdf_path or not pdf_path.is_file():
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=(
                f"Local filing PDF for entity '{entity}' / doc '{doc_name}' not found in "
                f"{pdf_dir}. Evidence-grounded verification requires access to SEC filings."
            ),
            claim=claim,
            parsed_candidates=[],
            citations=[],
            limitations=["Local filing corpus unavailable."],
        )


    # Extract pages from the targeted filing
    doc_links = load_doc_link_map(metadata_path)
    doc_url = doc_links.get(resolved_doc, "")

    pages: list[PageCorpusItem] = extract_pdf_pages(
        pdf_path=pdf_path,
        doc_name=resolved_doc,
        doc_link=doc_url,
    )

    if not pages:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=f"Could not extract any readable text pages from PDF: {pdf_path.name}.",
            claim=claim,
            parsed_candidates=[],
            citations=[],
            limitations=["PDF text extraction failed or yielded empty pages."],
        )

    # Build local lexical index over the filing pages
    retriever = BM25Retriever().fit(pages)
    query = f"{claim.metric} {claim.reporting_period} {claim.entity}"
    retrieval_results = retriever.retrieve(query, top_k=top_k)

    page_map = {p.page_id: p for p in pages}
    passages: list[EvidencePassage] = []

    for res in retrieval_results:
        full_page = page_map.get(res.page_id)
        if full_page:
            passages.append(
                EvidencePassage(
                    source_doc=res.doc_name,
                    page_index=res.page_index,
                    text=full_page.text,
                    source_url=full_page.doc_link or doc_url,
                    retrieval_score=res.score,
                    retrieval_rank=res.rank,
                    metadata={"char_count": full_page.char_count},
                )
            )

    # Pure deterministic verification
    return verify_claim(claim, passages)


def format_cli_output(result: VerificationResult) -> str:
    """Formats verification outcome for human-readable terminal display."""
    lines = []
    lines.append("=" * 78)
    lines.append(" FINVERITAS — EVIDENCE-GROUNDED FINANCIAL CLAIM VERIFICATION")
    lines.append("=" * 78)

    c = result.claim
    lines.append("\n[1] CLAIM SUMMARY")
    lines.append(f"  Entity:           {c.entity}")
    if c.doc_name:
        lines.append(f"  Target Document:  {c.doc_name}")
    lines.append(f"  Metric:           {c.metric} (canonical: {c.canonical_metric})")
    lines.append(f"  Reporting Period: {c.reporting_period} (normalized: {c.normalized_period})")
    lines.append(
        f"  Stated Value:     {c.value} {c.scale} ({c.canonical_currency} {c.canonical_value:,.2f})"
    )
    lines.append(f"  Tolerance:        {c.tolerance_pct * 100:.2f}%")

    lines.append("\n[2] VERIFICATION VERDICT")
    verdict_badge = f"*** {result.verdict.value} ***"
    lines.append(f"  Outcome:          {verdict_badge}")
    lines.append(f"  Rationale:        {result.explanation}")

    if result.citations:
        lines.append("\n[3] EVIDENCE CITATIONS")
        for idx, cite in enumerate(result.citations, start=1):
            lines.append(f"  Citation #{idx}:")
            lines.append(f"    Source Document:  {cite.source_doc}")
            lines.append(
                f"    PDF Page Index:   {cite.page_index} (Display Page: {cite.display_page})"
            )
            if cite.source_url:
                lines.append(f"    Source Link:      {cite.source_url}")
            lines.append(f'    Matched Excerpt:  "{cite.matched_text}"')

    if result.matched_candidate:
        cand = result.matched_candidate
        lines.append("\n[4] COMPARISON BREAKDOWN")
        cand_nom = f"{cand.nominal_value} {cand.scale}"
        filing_val_str = f"{cand_nom} ({cand.currency} {cand.canonical_value:,.2f})"
        lines.append(f"  Filing Value:     {filing_val_str}")
        claim_nom = f"{c.value} {c.scale}"
        claimed_val_str = f"{claim_nom} ({c.canonical_currency} {c.canonical_value:,.2f})"
        lines.append(f"  Claimed Value:    {claimed_val_str}")



        if cand.canonical_value != Decimal("0"):
            diff = c.canonical_value - cand.canonical_value
            pct_diff = (diff / abs(cand.canonical_value)) * 100
            lines.append(f"  Difference:       {diff:+,.2f} ({pct_diff:+.3f}%)")

    lines.append("\n[5] METHODOLOGICAL LIMITATIONS")
    for lim in result.limitations:
        lines.append(f"  - {lim}")
    lines.append("=" * 78)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a quantitative financial claim against SEC filing evidence."
    )
    parser.add_argument(
        "--claim",
        default=None,
        help="Optional text representation of the claim (for display and logging).",
    )
    parser.add_argument(
        "--entity",
        "--ticker",
        dest="entity",
        required=True,
        help="Target entity name or ticker (e.g. AMD, Boeing).",
    )
    parser.add_argument(
        "--metric", required=True, help="Financial metric (e.g. revenue, net income)."
    )
    parser.add_argument("--period", required=True, help="Reporting period (e.g. 2022, FY2022).")
    parser.add_argument("--value", required=True, help="Stated numerical value (e.g. 23.6, 23601).")
    parser.add_argument(
        "--scale",
        default="unit",
        help="Scale multiplier ('unit', 'thousand', 'million', 'billion'). Default: 'unit'.",
    )
    parser.add_argument("--currency", default="USD", help="Currency code. Default: 'USD'.")
    parser.add_argument(
        "--doc",
        dest="doc_name",
        default=None,
        help="Specific filing document name (e.g. AMD_2022_10K).",
    )
    parser.add_argument(
        "--tolerance",
        default="0.005",
        help="Relative verification tolerance (default: 0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/raw/financebench/pdfs",
        help="Directory containing source financial PDFs.",
    )
    parser.add_argument(
        "--doc-info",
        default="data/raw/financebench/financebench_document_information.jsonl",
        help="Document information JSONL metadata path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate pages to retrieve via BM25 (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON verification result.",
    )

    args = parser.parse_args()

    result = run_claim_verification(
        entity=args.entity,
        metric=args.metric,
        period=args.period,
        value=args.value,
        scale=args.scale,
        currency=args.currency,
        doc_name=args.doc_name,
        tolerance=args.tolerance,
        pdf_dir=Path(args.pdf_dir),
        metadata_path=Path(args.doc_info) if args.doc_info else None,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_cli_output(result))

    sys.exit(0)


if __name__ == "__main__":
    main()
