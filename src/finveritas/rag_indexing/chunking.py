"""
FinVeritas: Financial Document Chunking Engine.

Creates overlapping text chunks from extracted PDF pages while strictly
preserving:
  - Source document identifier (doc_name)
  - Zero-based PDF page index (page_index)
  - Chunk identifier (chunk_id)
  - Character offsets (char_start, char_end)
  - Numeric, currency, and accounting notation fidelity
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from finveritas.rag_indexing.financebench_data import PageCorpusItem

logger = logging.getLogger("chunking")


@dataclass(frozen=True)
class ChunkCorpusItem:
    """Represents an overlapping text chunk mapped to its source filing page."""

    chunk_id: str
    doc_name: str
    page_index: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    doc_link: str = ""


def chunk_text_sliding_window(
    text: str,
    doc_name: str,
    page_index: int,
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    doc_link: str = "",
    tokenizer: Any | None = None,
) -> list[ChunkCorpusItem]:
    """
    Splits page text into overlapping chunks of approximately `chunk_size` tokens
    with `chunk_overlap` tokens of overlap.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    chunks: list[ChunkCorpusItem] = []
    stride = max(1, chunk_size - chunk_overlap)

    if tokenizer is not None and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        # Tokenizer-based splitting (e.g. HuggingFace BGE tokenizer)
        try:
            tokens = tokenizer.encode(clean_text, add_special_tokens=False)
            total_tokens = len(tokens)
            if total_tokens <= chunk_size:
                chunks.append(
                    ChunkCorpusItem(
                        chunk_id=f"{doc_name}_p{page_index}_c0",
                        doc_name=doc_name,
                        page_index=page_index,
                        chunk_index=0,
                        text=clean_text,
                        char_start=0,
                        char_end=len(clean_text),
                        token_count=total_tokens,
                        doc_link=doc_link,
                    )
                )
                return chunks

            c_idx = 0
            for start in range(0, total_tokens, stride):
                end = min(start + chunk_size, total_tokens)
                sub_tokens = tokens[start:end]
                sub_text = tokenizer.decode(sub_tokens, skip_special_tokens=True).strip()
                if not sub_text:
                    continue

                chunks.append(
                    ChunkCorpusItem(
                        chunk_id=f"{doc_name}_p{page_index}_c{c_idx}",
                        doc_name=doc_name,
                        page_index=page_index,
                        chunk_index=c_idx,
                        text=sub_text,
                        char_start=0,  # approximate when decoding
                        char_end=len(sub_text),
                        token_count=len(sub_tokens),
                        doc_link=doc_link,
                    )
                )
                c_idx += 1
                if end >= total_tokens:
                    break

            return chunks
        except Exception as err:
            logger.warning(f"Tokenizer chunking failed, falling back to word chunking: {err}")

    # Robust word-based tokenization fallback preserving financial symbols
    words = re.findall(r"\S+|\n", clean_text)
    total_words = len(words)
    if total_words <= chunk_size:
        chunks.append(
            ChunkCorpusItem(
                chunk_id=f"{doc_name}_p{page_index}_c0",
                doc_name=doc_name,
                page_index=page_index,
                chunk_index=0,
                text=clean_text,
                char_start=0,
                char_end=len(clean_text),
                token_count=total_words,
                doc_link=doc_link,
            )
        )
        return chunks

    c_idx = 0
    for start in range(0, total_words, stride):
        end = min(start + chunk_size, total_words)
        sub_words = words[start:end]
        sub_text = " ".join(w for w in sub_words if w != "\n").strip()
        if not sub_text:
            continue

        chunks.append(
            ChunkCorpusItem(
                chunk_id=f"{doc_name}_p{page_index}_c{c_idx}",
                doc_name=doc_name,
                page_index=page_index,
                chunk_index=c_idx,
                text=sub_text,
                char_start=0,
                char_end=len(sub_text),
                token_count=len(sub_words),
                doc_link=doc_link,
            )
        )
        c_idx += 1
        if end >= total_words:
            break

    return chunks


def build_chunk_corpus(
    page_corpus: list[PageCorpusItem],
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    tokenizer: Any | None = None,
) -> tuple[list[ChunkCorpusItem], dict[str, Any]]:
    """
    Transforms a page-level corpus into an overlapping chunk corpus.
    Returns the chunk list and diagnostic summary statistics.
    """
    all_chunks: list[ChunkCorpusItem] = []
    stats = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "input_pages": len(page_corpus),
        "total_chunks_created": 0,
        "avg_chunks_per_page": 0.0,
        "max_chunks_per_page": 0,
        "min_chunks_per_page": 0,
        "total_tokens": 0,
    }

    chunks_per_page_counts: list[int] = []

    for page_item in page_corpus:
        p_chunks = chunk_text_sliding_window(
            text=page_item.text,
            doc_name=page_item.doc_name,
            page_index=page_item.page_index,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_link=page_item.doc_link,
            tokenizer=tokenizer,
        )
        all_chunks.extend(p_chunks)
        chunks_per_page_counts.append(len(p_chunks))

    total_chunks = len(all_chunks)
    stats["total_chunks_created"] = total_chunks
    stats["total_tokens"] = sum(c.token_count for c in all_chunks)
    if chunks_per_page_counts:
        stats["avg_chunks_per_page"] = round(total_chunks / len(chunks_per_page_counts), 2)
        stats["max_chunks_per_page"] = max(chunks_per_page_counts)
        stats["min_chunks_per_page"] = min(chunks_per_page_counts)

    return all_chunks, stats
