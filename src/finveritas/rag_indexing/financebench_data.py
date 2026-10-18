"""
FinVeritas: FinanceBench Dataset Acquisition, PDF Parsing & Corpus Builder.

Handles bounded downloading of the 150-question sample metadata and referenced
source PDFs from the official GitHub repository (patronus-ai/financebench),
conservative PDF page text extraction preserving financial notation, and
page-level corpus construction.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pypdf

logger = logging.getLogger("finveritas.rag_indexing.data")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
GITHUB_API_CONTENTS = "https://api.github.com/repos/patronus-ai/financebench/contents"

OPEN_SOURCE_METADATA_URL = f"{GITHUB_RAW_BASE}/data/financebench_open_source.jsonl"
DOC_INFO_METADATA_URL = f"{GITHUB_RAW_BASE}/data/financebench_document_information.jsonl"


@dataclasses.dataclass(frozen=True)
class PageCorpusItem:
    """Represents a single extracted PDF page in the retrieval corpus."""

    page_id: str  # Format: "{doc_name}_p{page_index}"
    doc_name: str  # e.g. "3M_2022_10K"
    page_index: int  # 0-based page index matching pypdf reader.pages
    text: str
    doc_link: str
    char_count: int


def normalize_financial_text(text: str) -> str:
    """
    Conservatively normalizes extracted financial PDF page text.
    Preserves numbers, signs (+/-), parentheses for negative figures, currency
    symbols ($/€/£/¥), decimals, percentages (%), and tabular line spacing.
    """
    if not text:
        return ""

    # NFKC unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace carriage returns and vertical tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\v", "\n")

    # Replace non-breaking and other exotic spaces with standard space
    text = re.sub(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]", " ", text)

    # Remove non-printable control characters except newline and tab
    text = "".join(ch for ch in text if ch in "\n\t" or (unicodedata.category(ch)[0] != "C"))

    # Collapse repeated horizontal whitespace within lines while preserving newlines
    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned)

    # Collapse excessive consecutive empty lines (max 2 newlines)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return result


def download_file_with_retry(
    url: str,
    target_path: Path,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> bool:
    """Downloads a remote file with bounded retries and exponential backoff."""
    if target_path.is_file() and target_path.stat().st_size > 0:
        logger.debug(f"Target file already exists: {target_path}")
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")

    headers = {"User-Agent": "FinVeritas-Research/0.1.0 (Academic Benchmark Audit)"}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {url} (attempt {attempt}/{max_retries})...")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
                temp_path.write_bytes(content)
            temp_path.replace(target_path)
            logger.info(f"Successfully downloaded {target_path.name} ({len(content)} bytes)")
            return True
        except (urllib.error.URLError, TimeoutError) as err:
            logger.warning(f"Download failed for {url} (attempt {attempt}): {err}")
            if temp_path.exists():
                temp_path.unlink()
            if attempt < max_retries:
                sleep_sec = backoff_factor**attempt
                time.sleep(sleep_sec)
        except Exception as err:
            logger.error(f"Unexpected error downloading {url}: {err}")
            if temp_path.exists():
                temp_path.unlink()
            break

    return False


def download_financebench_metadata(output_dir: Path) -> tuple[Path, Path]:
    """Downloads the two official FinanceBench metadata JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    open_source_path = output_dir / "financebench_open_source.jsonl"
    doc_info_path = output_dir / "financebench_document_information.jsonl"

    ok1 = download_file_with_retry(OPEN_SOURCE_METADATA_URL, open_source_path)
    ok2 = download_file_with_retry(DOC_INFO_METADATA_URL, doc_info_path)

    if not (ok1 and ok2):
        raise RuntimeError("Failed to acquire FinanceBench metadata JSONL files.")

    return open_source_path, doc_info_path


def load_financebench_metadata(
    open_source_path: Path,
    doc_info_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Loads and validates FinanceBench questions and document metadata."""
    if not open_source_path.is_file():
        raise FileNotFoundError(f"Missing open source metadata: {open_source_path}")

    questions = []
    with open(open_source_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            # Validation
            if "financebench_id" not in data or "question" not in data or "doc_name" not in data:
                raise ValueError(f"Line {line_num} in {open_source_path} missing required fields.")
            questions.append(data)

    doc_info_map = {}
    if doc_info_path and doc_info_path.is_file():
        with open(doc_info_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                info = json.loads(line_str)
                doc_name = info.get("doc_name")
                if doc_name:
                    doc_info_map[doc_name] = info

    return questions, doc_info_map


def download_financebench_pdfs(
    needed_doc_names: set[str],
    pdf_dir: Path,
    timeout: int = 30,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Downloads required financial source PDFs from patronus-ai/financebench.
    Returns status mapping with downloaded count, failed count, and failed doc list.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "required_count": len(needed_doc_names),
        "downloaded_count": 0,
        "existing_count": 0,
        "failed_count": 0,
        "failed_docs": [],
        "total_bytes": 0,
    }

    for doc_name in sorted(needed_doc_names):
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if pdf_path.is_file() and pdf_path.stat().st_size > 0:
            results["existing_count"] += 1
            results["total_bytes"] += pdf_path.stat().st_size
            continue

        raw_url = f"{GITHUB_RAW_BASE}/pdfs/{doc_name}.pdf"
        success = download_file_with_retry(
            url=raw_url,
            target_path=pdf_path,
            timeout=timeout,
            max_retries=max_retries,
        )
        if success:
            results["downloaded_count"] += 1
            results["total_bytes"] += pdf_path.stat().st_size
        else:
            results["failed_count"] += 1
            results["failed_docs"].append(doc_name)

    return results


def extract_pdf_pages(
    pdf_path: Path,
    doc_name: str,
    doc_link: str = "",
) -> list[PageCorpusItem]:
    """
    Extracts pages from a financial PDF using pypdf.
    Preserves 0-based page indexing aligned with FinanceBench evidence_page_num.
    """
    items: list[PageCorpusItem] = []
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as err:
        logger.error(f"Failed to read PDF {pdf_path}: {err}")
        return items

    for page_idx, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception as err:
            logger.warning(
                f"Error extracting text from {doc_name} page {page_idx}: {err}"
            )
            raw_text = ""

        cleaned_text = normalize_financial_text(raw_text)
        page_id = f"{doc_name}_p{page_idx}"

        items.append(
            PageCorpusItem(
                page_id=page_id,
                doc_name=doc_name,
                page_index=page_idx,
                text=cleaned_text,
                doc_link=doc_link,
                char_count=len(cleaned_text),
            )
        )

    return items


def build_page_corpus(
    pdf_dir: Path,
    doc_info_map: dict[str, dict[str, Any]] | None = None,
    needed_doc_names: set[str] | None = None,
) -> tuple[list[PageCorpusItem], dict[str, Any]]:
    """
    Builds the page-level retrieval corpus across all target PDFs.
    Returns the corpus items and diagnostic statistics.
    """
    corpus: list[PageCorpusItem] = []
    stats = {
        "total_pdfs_targeted": len(needed_doc_names) if needed_doc_names else 0,
        "pdfs_successfully_parsed": 0,
        "pdfs_failed": 0,
        "failed_pdf_names": [],
        "total_pages_extracted": 0,
        "empty_pages_count": 0,
        "non_empty_pages_count": 0,
    }

    doc_info_map = doc_info_map or {}
    target_docs = sorted(needed_doc_names) if needed_doc_names else [
        p.stem for p in pdf_dir.glob("*.pdf")
    ]

    for doc_name in target_docs:
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.is_file():
            stats["pdfs_failed"] += 1
            stats["failed_pdf_names"].append(doc_name)
            continue

        doc_info = doc_info_map.get(doc_name, {})
        doc_link = doc_info.get("doc_link", "")

        pages = extract_pdf_pages(pdf_path=pdf_path, doc_name=doc_name, doc_link=doc_link)
        if not pages:
            stats["pdfs_failed"] += 1
            stats["failed_pdf_names"].append(doc_name)
            continue

        stats["pdfs_successfully_parsed"] += 1
        for p in pages:
            corpus.append(p)
            stats["total_pages_extracted"] += 1
            if len(p.text.strip()) < 10:
                stats["empty_pages_count"] += 1
            else:
                stats["non_empty_pages_count"] += 1

    return corpus, stats
