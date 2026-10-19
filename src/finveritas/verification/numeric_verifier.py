"""
FinVeritas: Deterministic Numerical Financial Claim Verifier.

Provides structured data models and a pure, deterministic verification engine
that validates quantitative financial claims against SEC filing evidence passages
using Decimal arithmetic, explicit scale normalization, and documented rounding bounds.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Sequence


class VerificationVerdict(str, Enum):
    """Auditable verdict for financial claim verification."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# Canonical scale multipliers as exact Decimals
SCALE_MULTIPLIERS: dict[str, Decimal] = {
    "unit": Decimal("1"),
    "units": Decimal("1"),
    "one": Decimal("1"),
    "raw": Decimal("1"),
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "m": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "billions": Decimal("1000000000"),
    "b": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "trillions": Decimal("1000000000000"),
    "t": Decimal("1000000000000"),
}


def normalize_scale(scale: str | Decimal | int | None) -> Decimal:
    """Normalizes string, integer, or Decimal scale into an exact Decimal multiplier."""
    if scale is None:
        return Decimal("1")
    if isinstance(scale, (int, Decimal)):
        return Decimal(str(scale))
    s = scale.strip().lower()
    if s in SCALE_MULTIPLIERS:
        return SCALE_MULTIPLIERS[s]
    try:
        return Decimal(s)
    except InvalidOperation:
        raise ValueError(f"Unknown or unsupported scale representation: {scale}")


def canonical_scale_name(scale: str | Decimal | int | None) -> str:
    """Returns a canonical human-readable scale label."""
    mult = normalize_scale(scale)
    for name, m in SCALE_MULTIPLIERS.items():
        if m == mult and name in ("unit", "thousand", "million", "billion", "trillion"):
            return name
    return str(mult)


CURRENCY_ALIASES: dict[str, str] = {
    "usd": "USD",
    "$": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "eur": "EUR",
    "€": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "gbp": "GBP",
    "£": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "jpy": "JPY",
    "¥": "JPY",
    "yen": "JPY",
}


def normalize_currency(currency: str | None) -> str:
    """Normalizes currency identifiers to standard ISO-4217 uppercase codes."""
    if not currency:
        return "USD"
    c = currency.strip().lower()
    return CURRENCY_ALIASES.get(c, currency.strip().upper())


# Bounded set of supported financial metrics
SUPPORTED_METRICS: dict[str, list[str]] = {
    "REVENUE": [
        "revenue",
        "revenues",
        "total revenue",
        "total revenues",
        "net revenue",
        "net revenues",
        "total net revenue",
        "total net revenues",
        "net sales",
        "sales",
        "total net sales",
        "sales of products and services",
        "operating revenue",
        "operating revenues",
    ],
    "NET_INCOME": [
        "net income",
        "net earnings",
        "net income (loss)",
        "net income / (loss)",
        "net income/(loss)",
        "net loss",
        "net earnings (loss)",
        "net loss attributable to shareholders",
        "net income attributable to shareholders",
        "net income attributable to the company",
    ],
    "OPERATING_INCOME": [
        "operating income",
        "operating earnings",
        "operating profit",
        "operating income (loss)",
        "operating income/(loss)",
        "loss from operations",
        "operating profit (loss)",
        "income from operations",
    ],
    "TOTAL_ASSETS": [
        "total assets",
        "assets, total",
        "total assets and liabilities",
    ],
    "CAPITAL_EXPENDITURES": [
        "capital expenditures",
        "capital spending",
        "capex",
        "additions to property and equipment",
        "payments for property and equipment",
        "additions to property, plant and equipment",
        "purchase of property, plant and equipment",
        "purchases of property and equipment",
        "purchases of property, plant and equipment",
    ],
}


def canonicalize_metric(raw_metric: str) -> str | None:
    """
    Maps user or extracted metric descriptions to bounded canonical financial categories.
    Returns None if metric is outside the bounded MVP scope.
    """
    norm = unicodedata.normalize("NFKC", raw_metric).strip().lower()
    norm = re.sub(r"[^\w\s/()-]", "", norm)
    norm = re.sub(r"\s+", " ", norm)

    for canonical, aliases in SUPPORTED_METRICS.items():
        for alias in aliases:
            if norm == alias or norm.replace(" ", "") == alias.replace(" ", ""):
                return canonical
    return None


def normalize_period(period_str: str) -> str | None:
    """
    Standardizes fiscal year and quarter strings.
    E.g. 'FY2022', '2022', 'Fiscal 2022', 'December 31, 2022' -> '2022'.
    E.g. 'Q2 2023', '2023Q2', 'Second Quarter 2023' -> '2023Q2'.
    """
    if not period_str:
        return None
    s = period_str.strip()

    # Quarter pattern
    q_match = re.search(r"(?i)\bQ([1-4])\s*(?:of\s+)?(20\d\d)\b", s)
    if q_match:
        return f"{q_match.group(2)}Q{q_match.group(1)}"
    q_match2 = re.search(r"(?i)\b(20\d\d)\s*Q([1-4])\b", s)
    if q_match2:
        return f"{q_match2.group(1)}Q{q_match2.group(2)}"

    # Year pattern
    y_match = re.search(r"(?i)(?:FY|fiscal\s+year|fiscal)?\s*(20\d\d)\b", s)
    if y_match:
        return y_match.group(1)

    return s.upper()


@dataclasses.dataclass(frozen=True)
class FinancialClaim:
    """
    Structured representation of a numerical financial claim to be verified.

    Attributes:
        entity: Target organization (e.g. 'AMD', 'Boeing', 'AMD_2022_10K').
        metric: Stated financial metric (e.g. 'revenue', 'net income', 'operating income').
        reporting_period: Applicable fiscal period (e.g. '2022', 'FY2022', '2023Q2').
        value: Nominal reported number (e.g. Decimal('23.6') or Decimal('23600')).
        currency: Stated currency symbol/code (default: 'USD').
        scale: Scale of stated value (e.g. 'billion', 'million', 'thousand', 'unit').
        doc_name: Specific filing identifier if scoped (e.g. 'AMD_2022_10K').
        tolerance_pct: Relative tolerance for rounding verification (default 0.005 = 0.5%).
    """

    entity: str
    metric: str
    reporting_period: str
    value: Decimal
    currency: str = "USD"
    scale: str = "unit"
    doc_name: str | None = None
    tolerance_pct: Decimal = Decimal("0.005")

    def __post_init__(self) -> None:
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, "value", Decimal(str(self.value)))
        if not isinstance(self.tolerance_pct, Decimal):
            object.__setattr__(self, "tolerance_pct", Decimal(str(self.tolerance_pct)))

    @property
    def canonical_metric(self) -> str | None:
        return canonicalize_metric(self.metric)

    @property
    def normalized_period(self) -> str | None:
        return normalize_period(self.reporting_period)

    @property
    def canonical_currency(self) -> str:
        return normalize_currency(self.currency)

    @property
    def canonical_value(self) -> Decimal:
        """Returns the full unscaled value in base currency units."""
        multiplier = normalize_scale(self.scale)
        return self.value * multiplier

    @property
    def stated_decimal_places(self) -> int:
        """Counts decimal places in stated value (e.g. 23.6 -> 1, 23.60 -> 2)."""
        sign, digits, exponent = self.value.as_tuple()
        if exponent < 0:
            return -exponent
        return 0

    @property
    def rounding_bounds(self) -> tuple[Decimal, Decimal]:
        """
        Calculates the half-step rounding interval implied by the stated decimal precision.
        For example: 23.6 billion -> [23.55e9, 23.65e9).
        """
        multiplier = normalize_scale(self.scale)
        d = self.stated_decimal_places
        half_step = (Decimal("10") ** (-d)) / Decimal("2")
        lower_nom = self.value - half_step
        upper_nom = self.value + half_step
        return (lower_nom * multiplier, upper_nom * multiplier)


@dataclasses.dataclass(frozen=True)
class EvidencePassage:
    """
    Represents an evidence text passage retrieved from a filing.

    Attributes:
        source_doc: Name of the filing document (e.g. 'AMD_2022_10K').
        page_index: 0-based PDF page index (aligned with FinanceBench evidence_page_num).
        text: Complete extracted passage text (untruncated).
        source_url: Link to original filing if available.
        retrieval_score: Score from retrieval engine.
        retrieval_rank: Rank position in retrieval results (1-based).
        metadata: Additional metadata dictionary.
    """

    source_doc: str
    page_index: int
    text: str
    source_url: str = ""
    retrieval_score: float = 0.0
    retrieval_rank: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ParsedEvidenceCandidate:
    """A numerical candidate extracted from evidence text."""

    source_doc: str
    page_index: int
    raw_metric_text: str
    canonical_metric: str
    reporting_period: str
    nominal_value: Decimal
    scale: str
    canonical_value: Decimal
    currency: str
    is_negative: bool
    matched_text: str
    source_url: str = ""


@dataclasses.dataclass(frozen=True)
class EvidenceCitation:
    """Structured citation referencing filing evidence."""

    source_doc: str
    page_index: int  # 0-based
    display_page: int  # 1-based (page_index + 1)
    matched_text: str
    source_url: str = ""


@dataclasses.dataclass(frozen=True)
class VerificationResult:
    """
    Complete audit outcome of numerical claim verification.
    """

    verdict: VerificationVerdict
    explanation: str
    claim: FinancialClaim
    parsed_candidates: list[ParsedEvidenceCandidate]
    citations: list[EvidenceCitation]
    limitations: list[str]
    matched_candidate: ParsedEvidenceCandidate | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "explanation": self.explanation,
            "claim": {
                "entity": self.claim.entity,
                "metric": self.claim.metric,
                "canonical_metric": self.claim.canonical_metric,
                "reporting_period": self.claim.reporting_period,
                "normalized_period": self.claim.normalized_period,
                "stated_value": str(self.claim.value),
                "scale": self.claim.scale,
                "canonical_value": str(self.claim.canonical_value),
                "currency": self.claim.canonical_currency,
                "tolerance_pct": str(self.claim.tolerance_pct),
            },
            "citations": [
                {
                    "source_doc": c.source_doc,
                    "page_index": c.page_index,
                    "display_page": c.display_page,
                    "matched_text": c.matched_text,
                    "source_url": c.source_url,
                }
                for c in self.citations
            ],
            "limitations": self.limitations,
            "candidate_count": len(self.parsed_candidates),
        }


# ---------------------------------------------------------------------------
# Robust Extraction Engine for Tables and Narrative Sentences
# ---------------------------------------------------------------------------

SCALE_HEADER_RE = re.compile(
    r"\(\s*(?:in|amounts?\s+in|dollars?\s+in)\s+(billions?|millions?|thousands?)\b",
    re.IGNORECASE,
)

CURRENCY_HEADER_RE = re.compile(
    r"\b(?:dollars|usd|\$|euros?|eur|€|pounds?|gbp|£)\b",
    re.IGNORECASE,
)

# Number pattern handling accounting negative parentheses: e.g. (5,053), ($4,935), -12.4
NUMBER_PATTERN = (
    r"(?:\(?\s*[$€£]?\s*\(?\s*"
    r"([+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[+-]?[0-9]+(?:\.[0-9]+)?)"
    r"\s*\)?\s*\)?)"
)


def parse_numeric_token(tok: str) -> tuple[Decimal, bool] | None:
    """Parses a financial number token, handling accounting parentheses for negative values."""
    s = tok.strip()
    if not s:
        return None

    is_negative = False
    # Check accounting parentheses: e.g. (1,234) or ($1,234) or ( 1,234 )
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("($") and s.endswith(")")):
        is_negative = True
        s = s.strip("()$ ")

    if s.startswith("-"):
        is_negative = True
        s = s.lstrip("-")

    # Clean remaining currency symbols or whitespace
    s = s.replace("$", "").replace("€", "").replace("£", "").replace(",", "").strip()

    try:
        val = Decimal(s)
        if is_negative:
            val = -val
        return val, is_negative
    except InvalidOperation:
        return None


PROSE_INDICATORS = (
    " was ",
    " were ",
    " increase ",
    " decrease ",
    " compared to ",
    " primarily ",
    " because ",
    " due to ",
    " driven by ",
    " consisted of ",
    " table of contents",
    " forward-looking",
)


def extract_table_candidates(passage: EvidencePassage) -> list[ParsedEvidenceCandidate]:
    """
    Extracts structured candidates from multi-column financial statement tables.
    Identifies column year headers (e.g. 2022, 2021, 2020), table scale (e.g. millions),
    and aligns row figures to their respective reporting periods.
    """
    candidates: list[ParsedEvidenceCandidate] = []
    lines = [line.strip() for line in passage.text.split("\n") if line.strip()]

    # 1. Detect page-level / table-level scale header
    default_scale = "unit"
    for line in lines[:25]:
        m = SCALE_HEADER_RE.search(line)
        if m:
            default_scale = canonical_scale_name(m.group(1))
            break

    # 2. Detect default currency
    default_curr = "USD"
    for line in lines[:25]:
        if "euro" in line.lower() or "€" in line:
            default_curr = "EUR"
            break
        if "pound" in line.lower() or "£" in line:
            default_curr = "GBP"
            break

    # 3. Detect column year headers in tabular header rows
    col_years: list[str] = []
    header_line_idx = -1

    for idx, line in enumerate(lines[:35]):
        # Header rows must be concise tabular rows, not narrative sentences
        line_lower = f" {line.lower()} "
        if any(p in line_lower for p in PROSE_INDICATORS):
            continue
        words = line.split()
        if len(words) > 15:
            continue

        years = re.findall(r"\b(20\d\d)\b", line)
        if len(years) >= 2:
            col_years = years
            header_line_idx = idx
            break

    # If column headers found, scan following lines for financial metric rows
    if col_years and header_line_idx >= 0:
        for line in lines[header_line_idx + 1 :]:
            # Stop table parsing if encountering next major section
            if line.startswith("ITEM ") or line.startswith("PART ") or "Table of Contents" in line:
                break

            line_lower = f" {line.lower()} "
            if any(p in line_lower for p in PROSE_INDICATORS):
                continue

            # Check if line begins with or contains a supported metric
            matched_canonical: str | None = None
            matched_alias: str | None = None

            for canonical, aliases in SUPPORTED_METRICS.items():
                for alias in aliases:
                    pattern = rf"(?i)\b{re.escape(alias)}\b"
                    if re.search(pattern, line):
                        matched_canonical = canonical
                        matched_alias = alias
                        break
                if matched_canonical:
                    break

            if not matched_canonical:
                continue

            # If matched REVENUE, ensure it is not Cost of Sales or Cost of Revenue
            if matched_canonical == "REVENUE" and re.search(r"(?i)\bcost\s+of\b", line):
                continue

            # Check if metric row indicates net loss
            is_inherently_loss = "loss" in line.lower() and "income" not in line.lower()

            # Extract numbers from row
            num_matches = list(
                re.finditer(
                    r"(?:\(?\s*[$€£]?\s*\(?\s*[+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\s*\)?\s*\)?)",
                    line,
                )
            )

            valid_nums = []
            for nm in num_matches:
                raw_num = nm.group(0).strip()
                if not any(c.isdigit() for c in raw_num):
                    continue
                # Skip if the number token is identical to a column year (e.g. 2022)
                clean_digits = (
                    raw_num.replace(",", "")
                    .replace("$", "")
                    .replace("(", "")
                    .replace(")", "")
                    .strip()
                )
                if clean_digits in col_years:
                    continue

                parsed = parse_numeric_token(raw_num)
                if parsed is not None:
                    valid_nums.append((raw_num, parsed[0], parsed[1]))

            # Pair row numbers with column years
            if valid_nums:
                for col_idx, (raw_str, val, is_neg) in enumerate(valid_nums[: len(col_years)]):
                    period = col_years[col_idx]
                    eff_val = val
                    eff_neg = is_neg
                    if is_inherently_loss and eff_val > 0:
                        eff_val = -eff_val
                        eff_neg = True

                    mult = normalize_scale(default_scale)
                    canon_val = eff_val * mult

                    candidates.append(
                        ParsedEvidenceCandidate(
                            source_doc=passage.source_doc,
                            page_index=passage.page_index,
                            raw_metric_text=matched_alias or line[:40],
                            canonical_metric=matched_canonical,
                            reporting_period=period,
                            nominal_value=eff_val,
                            scale=default_scale,
                            canonical_value=canon_val,
                            currency=default_curr,
                            is_negative=eff_neg,
                            matched_text=line,
                            source_url=passage.source_url,
                        )
                    )

    return candidates


def extract_narrative_candidates(passage: EvidencePassage) -> list[ParsedEvidenceCandidate]:
    """
    Extracts structured candidates from narrative sentences in MD&A and footnotes.
    E.g. 'revenue for 2022 was $23.6 billion, compared to 2021 net revenue of $16.4B'
    """
    candidates: list[ParsedEvidenceCandidate] = []
    text = passage.text

    # Narrative pattern 1: [metric] ... [period] ... was/of [$] [number] [scale]
    for canonical, aliases in SUPPORTED_METRICS.items():
        for alias in aliases:
            pattern = re.compile(
                rf"(?i)\b{re.escape(alias)}\b[^.\n]{{0,60}}?\b(?:for|in|of)?\s*(?:fiscal\s+)?\b(20\d\d)\b[^.\n]{{0,40}}?\b(?:was|of|were|reached|totaled)\s+([$€£]?)\s*([0-9]{{1,3}}(?:,[0-9]{{3}})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(billions?|millions?|thousands?)?",
            )
            for m in pattern.finditer(text):
                period = m.group(1)
                curr_sym = m.group(2)
                raw_val = m.group(3)
                raw_scale = m.group(4) or "unit"

                # Check if followed by % (growth rate / margin, not dollar figure)
                after_text = text[m.end() : m.end() + 10].lstrip()
                if after_text.startswith("%") or after_text.lower().startswith("percent"):
                    continue

                # Narrative figures require currency indicator or explicit scale
                if not curr_sym and (not m.group(4) or raw_scale == "unit"):
                    continue

                parsed = parse_numeric_token(raw_val)
                if parsed is None:
                    continue

                nom_val, is_neg = parsed
                scale_norm = canonical_scale_name(raw_scale)
                mult = normalize_scale(scale_norm)
                curr_norm = normalize_currency(curr_sym) if curr_sym else "USD"

                matched_sentence = (
                    text[max(0, m.start() - 15) : min(len(text), m.end() + 15)]
                    .replace("\n", " ")
                    .strip()
                )
                candidates.append(
                    ParsedEvidenceCandidate(
                        source_doc=passage.source_doc,
                        page_index=passage.page_index,
                        raw_metric_text=alias,
                        canonical_metric=canonical,
                        reporting_period=period,
                        nominal_value=nom_val,
                        scale=scale_norm,
                        canonical_value=nom_val * mult,
                        currency=curr_norm,
                        is_negative=is_neg,
                        matched_text=matched_sentence,
                        source_url=passage.source_url,
                    )
                )

            # Narrative pattern 2: [$] [number] [scale] in/of [metric] for [period]
            pattern2 = re.compile(
                rf"(?i)([$€£]?)\s*([0-9]{{1,3}}(?:,[0-9]{{3}})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(billions?|millions?|thousands?)?\s+(?:in|of)\s+\b{re.escape(alias)}\b[^.\n]{{0,40}}?\b(?:for|in)\s*(?:fiscal\s+)?\b(20\d\d)\b",
            )
            for m in pattern2.finditer(text):
                curr_sym = m.group(1)
                raw_val = m.group(2)
                raw_scale = m.group(3) or "unit"
                period = m.group(4)

                after_text = text[m.end() : m.end() + 10].lstrip()
                if after_text.startswith("%") or after_text.lower().startswith("percent"):
                    continue

                if not curr_sym and (not m.group(3) or raw_scale == "unit"):
                    continue

                parsed = parse_numeric_token(raw_val)
                if parsed is None:
                    continue

                nom_val, is_neg = parsed
                scale_norm = canonical_scale_name(raw_scale)
                mult = normalize_scale(scale_norm)
                curr_norm = normalize_currency(curr_sym) if curr_sym else "USD"

                matched_sentence = (
                    text[max(0, m.start() - 15) : min(len(text), m.end() + 15)]
                    .replace("\n", " ")
                    .strip()
                )
                candidates.append(
                    ParsedEvidenceCandidate(
                        source_doc=passage.source_doc,
                        page_index=passage.page_index,
                        raw_metric_text=alias,
                        canonical_metric=canonical,
                        reporting_period=period,
                        nominal_value=nom_val,
                        scale=scale_norm,
                        canonical_value=nom_val * mult,
                        currency=curr_norm,
                        is_negative=is_neg,
                        matched_text=matched_sentence,
                        source_url=passage.source_url,
                    )
                )

    return candidates


def parse_evidence_candidates(passage: EvidencePassage) -> list[ParsedEvidenceCandidate]:
    """Combines table-based and narrative extraction into a unified candidate pool."""
    all_cands: list[ParsedEvidenceCandidate] = []
    all_cands.extend(extract_table_candidates(passage))
    all_cands.extend(extract_narrative_candidates(passage))

    # Deduplicate exact candidate matches (same doc, page, metric, period, canonical_value)
    unique_map: dict[tuple[str, int, str, str, Decimal], ParsedEvidenceCandidate] = {}
    for c in all_cands:
        key = (
            c.source_doc,
            c.page_index,
            c.canonical_metric,
            c.reporting_period,
            c.canonical_value,
        )
        if key not in unique_map:
            unique_map[key] = c

    return list(unique_map.values())


# ---------------------------------------------------------------------------
# Pure Verification Engine
# ---------------------------------------------------------------------------


def verify_claim(
    claim: FinancialClaim,
    evidence: Sequence[EvidencePassage],
) -> VerificationResult:
    """
    Pure verification function that evaluates a structured financial claim against
    retrieved filing evidence passages.

    Rules:
    - SUPPORTED: Relevant filing evidence unambiguously reports the same metric and period,
      and its value matches the claim within explicit decimal rounding bounds or relative tolerance.
    - CONTRADICTED: Directly comparable, unambiguous filing evidence reports a materially
      conflicting value for the same metric and period.
    - INSUFFICIENT_EVIDENCE: Missing evidence, parsing ambiguity, unmatched periods,
      currency/unit incompatibilities, conflicting candidates, or unsupported metric.

    Never infers truth from retrieval similarity alone.
    Never marks a claim CONTRADICTED merely because the value fails to appear.
    Fails closed with INSUFFICIENT_EVIDENCE when ambiguous.
    """
    limitations: list[str] = [
        "Scope bounded to revenue, net income, operating income, total assets, and capex.",
        "Deterministic heuristic parsing without non-GAAP reconciliation or visual layout OCR.",
        "Requires explicit reporting period and currency denomination alignment.",
    ]

    # Gate 1: Check claim metric support
    canonical_metric = claim.canonical_metric
    if not canonical_metric:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=(
                f"Unsupported metric: '{claim.metric}'. "
                "Only bounded core financial metrics are supported."
            ),
            claim=claim,
            parsed_candidates=[],
            citations=[],
            limitations=limitations,
        )

    # Gate 2: Check claim period
    norm_period = claim.normalized_period
    if not norm_period:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=f"Unrecognized reporting period: '{claim.reporting_period}'.",
            claim=claim,
            parsed_candidates=[],
            citations=[],
            limitations=limitations,
        )

    # Gate 3: Check evidence availability
    if not evidence:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation="No evidence passages provided for verification.",
            claim=claim,
            parsed_candidates=[],
            citations=[],
            limitations=limitations,
        )

    # Gate 4: Filter passages matching entity / document scope
    target_entity = claim.entity.strip().lower()
    target_doc = claim.doc_name.strip().lower() if claim.doc_name else None

    scoped_passages: list[EvidencePassage] = []
    for p in evidence:
        doc_lower = p.source_doc.lower()
        if target_doc and target_doc not in doc_lower:
            continue
        if not target_doc and target_entity not in doc_lower:
            continue
        scoped_passages.append(p)

    if not scoped_passages:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=(
                f"No evidence passages matched entity '{claim.entity}' or doc '{claim.doc_name}'. "
                f"Retrieved passages were from: {sorted(set(p.source_doc for p in evidence))}."
            ),
            claim=claim,
            parsed_candidates=[],
            citations=[],
            limitations=limitations,
        )

    # Gate 5: Parse candidates from scoped passages
    all_candidates: list[ParsedEvidenceCandidate] = []
    for p in scoped_passages:
        cands = parse_evidence_candidates(p)
        all_candidates.extend(cands)

    if not all_candidates:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=(
                f"No structured financial values for metric {canonical_metric} could be "
                f"unambiguously parsed from the {len(scoped_passages)} retrieved passages."
            ),
            claim=claim,
            parsed_candidates=[],
            citations=[],
            limitations=limitations,
        )

    # Gate 6: Filter candidates matching target metric and period
    metric_candidates = [c for c in all_candidates if c.canonical_metric == canonical_metric]
    if not metric_candidates:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=f"No parsed evidence mentions the target metric '{canonical_metric}'.",
            claim=claim,
            parsed_candidates=all_candidates,
            citations=[],
            limitations=limitations,
        )

    period_candidates = [
        c for c in metric_candidates if normalize_period(c.reporting_period) == norm_period
    ]
    if not period_candidates:
        found_periods = sorted(set(c.reporting_period for c in metric_candidates))
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=(
                f"Reporting period mismatch: claim specifies '{norm_period}', but evidence "
                f"only reports periods {found_periods} for metric {canonical_metric}."
            ),
            claim=claim,
            parsed_candidates=metric_candidates,
            citations=[],
            limitations=limitations,
        )

    # Gate 7: Check currency compatibility
    claim_curr = claim.canonical_currency
    curr_mismatches = [c for c in period_candidates if c.currency != claim_curr]
    matching_curr_cands = [c for c in period_candidates if c.currency == claim_curr]

    if not matching_curr_cands and curr_mismatches:
        ev_currs = sorted(set(c.currency for c in curr_mismatches))
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=(
                f"Currency mismatch: claim is denominated in {claim_curr}, but filing evidence "
                f"is denominated in {ev_currs}."
            ),
            claim=claim,
            parsed_candidates=period_candidates,
            citations=[],
            limitations=limitations,
        )

    # Gate 8: Disambiguate multiple candidates
    # Group candidates by distinct canonical value within rounding tolerance
    unique_values: list[Decimal] = []
    for c in matching_curr_cands:
        val = c.canonical_value
        if not any(
            abs(val - uv) / max(abs(uv), Decimal("1")) <= Decimal("0.005") for uv in unique_values
        ):
            unique_values.append(val)

    if len(unique_values) > 1:
        # Multiple conflicting candidates found for the same metric, period, and currency!
        val_strs = [f"{c.currency} {c.canonical_value:,.2f}" for c in matching_curr_cands[:4]]
        conf_expl = (
            f"Multiple conflicting evidence figures found for {canonical_metric} in period "
            f"{norm_period}: {', '.join(val_strs)}. Cannot verify without additional context."
        )
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
            explanation=conf_expl,
            claim=claim,
            parsed_candidates=matching_curr_cands,
            citations=[],
            limitations=limitations,
        )

    # Gate 9: Compare claim canonical value against evidence value
    best_candidate = matching_curr_cands[0]
    claim_val = claim.canonical_value
    evidence_val = best_candidate.canonical_value

    # Evaluate rounding bounds and relative difference
    lower_bound, upper_bound = claim.rounding_bounds
    within_rounding_interval = lower_bound <= evidence_val <= upper_bound

    if evidence_val != Decimal("0"):
        rel_diff = abs(claim_val - evidence_val) / abs(evidence_val)
    else:
        rel_diff = abs(claim_val - evidence_val)

    within_relative_tolerance = rel_diff <= claim.tolerance_pct

    # Prepare citation
    citation = EvidenceCitation(
        source_doc=best_candidate.source_doc,
        page_index=best_candidate.page_index,
        display_page=best_candidate.page_index + 1,
        matched_text=best_candidate.matched_text,
        source_url=best_candidate.source_url,
    )

    if within_rounding_interval or within_relative_tolerance:
        # Supported!
        cand_str = f"{best_candidate.nominal_value:,.2f} {best_candidate.scale}"
        ev_str = f"{claim.canonical_currency} {evidence_val:,.2f}"
        claim_str = f"{claim.value} {claim.scale} ({claim.canonical_currency} {claim_val:,.2f})"
        diff_str = f"{rel_diff * 100:.3f}% <= {claim.tolerance_pct * 100:.2f}%"
        explanation = (
            f"Filing '{best_candidate.source_doc}' page {best_candidate.page_index} "
            f"(display page {citation.display_page}) reports {canonical_metric} for period "
            f"{norm_period} as {cand_str} ({ev_str}), which matches the claimed value of "
            f"{claim_str} within documented tolerance (relative difference: {diff_str})."
        )
        return VerificationResult(
            verdict=VerificationVerdict.SUPPORTED,
            explanation=explanation,
            claim=claim,
            parsed_candidates=matching_curr_cands,
            citations=[citation],
            limitations=limitations,
            matched_candidate=best_candidate,
        )
    else:
        # Materially contradicted!
        diff_pct = ((claim_val - evidence_val) / abs(evidence_val)) * 100
        sign_str = f"+{diff_pct:.2f}%" if diff_pct > 0 else f"{diff_pct:.2f}%"
        cand_str = f"{best_candidate.nominal_value:,.2f} {best_candidate.scale}"
        ev_str = f"{claim.canonical_currency} {evidence_val:,.2f}"
        claim_str = f"{claim.value} {claim.scale} ({claim.canonical_currency} {claim_val:,.2f})"
        tol_str = f"{claim.tolerance_pct * 100:.2f}%"
        explanation = (
            f"Filing '{best_candidate.source_doc}' page {best_candidate.page_index} "
            f"(display page {citation.display_page}) unambiguously reports {canonical_metric} "
            f"for period {norm_period} as {cand_str} ({ev_str}), materially contradicting "
            f"the claimed value of {claim_str} by {sign_str} (exceeds tolerance of {tol_str})."
        )
        return VerificationResult(
            verdict=VerificationVerdict.CONTRADICTED,
            explanation=explanation,
            claim=claim,
            parsed_candidates=matching_curr_cands,
            citations=[citation],
            limitations=limitations,
            matched_candidate=best_candidate,
        )
