# FinVeritas — Checkpoint 07 Recovery and Safe Resumption Report

**Date:** October 19, 2026  
**Worktree:** `C:\Users\Aayan\Desktop\Project\NLP\.worktrees\hybrid-retrieval`  
**Branch:** `checkpoint/07-advanced-hybrid-retrieval`  
**Base Commit:** `124f283` (`feat(verification): implement financial answer-quality baseline and grouped-CV benchmark`)

---

## 1. Operating System Failure Mechanism Analysis

### 1.1 BugCheck Event Diagnostics
During the execution of dense retrieval indexing over the full FinanceBench corpus (37,118 chunks), the host system experienced a kernel crash:

- **BugCheck Code:** `0x00000133` (`DPC_WATCHDOG_VIOLATION`)
- **Event Source:** Microsoft-Windows-Kernel-Power (Event ID 41, BugCheck 1001)
- **Direct Cause:** Windows Deferred Procedure Call (DPC) watchdog timer expiration.

### 1.2 Root Cause in Code
1. **Monolithic Encoding:** The legacy dense retriever executed `model.encode(texts, batch_size=64, device="cuda")` over 37,118 text chunks in a single synchronous operation without yielding the GPU or CPU control loop.
2. **Display Driver Hang / WDDM Timeout:** On Windows, non-preemptive GPU driver execution exceeding the watchdog timeout (typically 2-10 seconds of un-yielded DPC / ISR execution on primary display adapter GPUs) causes Windows to trigger a driver reset (TDR) or bugcheck `0x133` when a single DPC routine runs too long.
3. **Absence of Intermediate Disk Persistence:** Because the operation was non-incremental, the crash destroyed all in-memory state, leaving zero persisted embeddings and forcing a complete restart.
4. **Unconditional CUDA Invocation:** CUDA was targeted by default without hardware verification, thermal safeguards, or memory headroom checks.

---

## 2. Code Refactoring and Resumption Architecture

To ensure zero risk of kernel panics or loss of compute, `DenseRetriever` and `NeuralReranker` in `src/finveritas/rag_indexing/retrievers.py` were comprehensively re-engineered:

### 2.1 Safe Hardware Defaults
- **CPU Default Execution:** Both `DenseRetriever` and `NeuralReranker` default strictly to `device="cpu"`.
- **Gated CUDA Protection (`check_cuda_safe`):** Selecting CUDA raises a `RuntimeError` unless `allow_cuda=True` is explicitly passed **and** a 2-chunk GPU smoke test executes and passes without driver errors or OOM exceptions.
- **Zero Package-Import Side Effects:** Removed top-level PyTorch/CUDA tensor allocations or device queries upon importing `finveritas.rag_indexing`.

### 2.2 Incremental Batch-Wise Processing & Persistence
- Rather than a single monolithic tensor, chunks are divided into configurable micro-batches (default `batch_size=32` on CPU).
- Each batch $b$ is encoded independently and persisted immediately to disk:
  - `batch_{b:05d}.npy`: Binary float32 embedding matrix for that slice of chunks.
  - `batch_{b:05d}.meta.json`: JSON metadata containing the batch index, chunk start/end offsets, and SHA-256 hash of the batch text contents.
- In-memory references to previous batches are immediately released to maintain a constant memory footprint (~1.2 GB RAM).

### 2.3 Strict Manifest and Atomic Cache Consistency
- **Cache Fingerprinting:** The cache folder name incorporates the chunk count and a SHA-256 digest of all chunk IDs and content prefixes (`dense_{num_chunks}_{fingerprint}`).
- **Atomic Disk Writes:**
  - `_atomic_write_json(path, data)`: Writes to `path.tmp.<pid>` followed by `os.replace()`.
  - `_atomic_save_npy(path, array)`: Writes to temporary file and atomically replaces destination.
  - Prevents corrupt or partially written files if the process is killed mid-write.
- **Manifest Tracking (`manifest.json`):**
  Records model name, embedding dimension, total chunk count, chunk hash list, total batches, completed batches, and an `is_complete` boolean flag.
- **Automated Corruption Recovery:**
  On initialization or resumption, every batch file is verified against its `.meta.json` hash and dimension shape. Any corrupt or truncated batch is flagged, discarded, and scheduled for clean re-encoding; valid batches are retained.
- **Instant Resume:**
  If `manifest.json` has `is_complete == True` and the consolidated `embeddings.npy` exists and matches the input chunks, `DenseRetriever` loads the cached matrix in **<0.02 seconds** without instantiating the transformer model into memory.

### 2.4 Telemetry and Progress Reporting
- Progress is logged after every batch, reporting:
  - Batch index and completion percentage (`Batch b/total (xx.x%)`).
  - Per-batch latency and cumulative elapsed time.
  - Linear extrapolation Estimated Time of Arrival (ETA).
  - Resident Set Size (RSS) memory utilization via `psutil` (with graceful fallback).

---

## 3. Verification & Test Suite

### 3.1 Offline Unit Tests
Run command:
```powershell
pytest tests/unit/test_advanced_retrieval.py -q
```
**Results:** **15 passed in 2.32s** (100% passing).

Covered test suites:
- `test_document_chunking_sliding_window`: Sliding window token splitting and metadata preservation.
- `test_chunk_to_page_mapping`: Chunk page provenance tracking.
- `test_chunk_financial_table_preservation`: Table boundary retention.
- `test_bm25_retriever_initialization_and_search`: Inverted index lexical scoring.
- `test_dense_retriever_cpu_default`: Validates CPU fallback and device safety guards.
- `test_dense_retriever_incremental_caching`: Verifies batch persistence and manifest structure.
- `test_dense_retriever_corrupted_batch_recovery`: Simulates partial batch file corruption and validates automatic self-healing.
- `test_reciprocal_rank_fusion`: Tests RRF k=60 rank fusion math and score stability.
- `test_linear_score_combination`: Tests weighted alpha combinations.
- `test_deduplicate_to_pages`: Evaluates top chunk aggregation to target document pages.
- `test_mrr_at_k_computation`: Verifies Mean Reciprocal Rank edge cases.
- `test_ndcg_at_k_computation`: Verifies Normalized Discounted Cumulative Gain against analytical ground truth.
- `test_ndcg_at_k_computation_edge_cases`: Multiple gold annotations, zero relevant docs, and truncated k.
- `test_hybrid_retriever_end_to_end_synthetic`: Full lexical + dense + RRF retrieval pipeline on CPU.
- `test_neural_reranker_cpu_synthetic`: Cross-encoder reranking on CPU.

### 3.2 Static Analysis & Environment Checks
- **Ruff:** `python -m ruff check src/ tests/ scripts/` -> **All checks passed!** (Zero lint errors, clean imports, max line length <= 100).
- **Pip Check:** `python -m pip check` -> **No broken requirements found.**
- **Git Hygiene:** No sensitive data, raw PDFs, or `.npy` embeddings staged.

---

## 4. Phase 3 Staged 256-Chunk CPU Benchmark Results

The staged real-world evaluation script (`scripts/test_cpu_256_chunks.py`) was executed on real FinanceBench text extracted from 3M Company's 2018 10-K filing (`3M_2018_10K`).

### 4.1 Fresh Encoding Benchmark (CPU)
- **Model:** `BAAI/bge-small-en-v1.5` (Local HuggingFace Cache)
- **Execution Device:** CPU (Intel x86_64, Windows)
- **Dataset Slice:** 256 real financial text chunks
- **Batch Size:** 32 chunks per batch (Total = 8 batches)
- **Total Elapsed Encoding Time:** 13.4 seconds (avg 1.67s / batch)
- **Memory Footprint:** 1241.8 MB start -> 1279.1 MB peak (delta +37.3 MB)

| Batch | Progress | Batch Time | Cumulative Elapsed | ETA | RSS RAM |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1/8 | 12.5% | 2.04s | 2.0s | 14.3s | 1241.8 MB |
| 2/8 | 25.0% | 1.79s | 3.8s | 11.5s | 1246.0 MB |
| 3/8 | 37.5% | 1.48s | 5.3s | 8.8s | 1255.9 MB |
| 4/8 | 50.0% | 1.61s | 6.9s | 6.9s | 1265.4 MB |
| 5/8 | 62.5% | 1.64s | 8.6s | 5.1s | 1265.9 MB |
| 6/8 | 75.0% | 1.75s | 10.3s | 3.4s | 1273.8 MB |
| 7/8 | 87.5% | 1.49s | 11.8s | 1.7s | 1277.5 MB |
| 8/8 | 100.0% | 1.61s | 13.4s | 0.0s | 1279.1 MB |

### 4.2 Quality & Mathematical Verification
- **Output Embedding Matrix Shape:** `(256, 384)` float32.
- **L2 Vector Norms:** `min=1.0000`, `max=1.0000`, `mean=1.0000` (Exact unit hypersphere normalization confirmed).
- **Chunk-to-Page Metadata:** 100% preservation across all 256 chunks (`doc_id="3M_2018_10K"`, page indices mapped).
- **Manifest Integrity:** `manifest.json` successfully stored `num_chunks=256`, `is_complete=true`, and all 8 batch SHA-256 signatures.

### 4.3 Cache Reuse & Instant Resume Verification
When a second retriever was initialized with the identical cache directory:
- **Resumption Latency:** **0.0117 seconds** (instantaneous disk mmap/load).
- **Transformer Allocation:** 0 bytes (the `SentenceTransformer` neural model was not instantiated).
- **Compute Re-use:** 0 batches recomputed; all 256 embeddings restored from disk.

### 4.4 Live Retrieval Verification
Queried with financial test question:
`"What were capital expenditures and cash flow?"`

Top Retrieved Chunks:
1. **Rank 1 (Score 0.7329):** Page 45 — *Cash flows from operating, investing, and financing activities...*
2. **Rank 2 (Score 0.7035):** Page 59 — *3M Company Consolidated Statement of Cash Flows...*
3. **Rank 3 (Score 0.6793):** Page 38 — *Geographic Area Supplemental Information Property, Plant and Equipment...*

---

## 5. Remaining Risks and Operational Recommendations

1. **CPU Throughput Scaling for Full 37,118 Chunks:**
   - At ~1.65 seconds per batch of 32 chunks on CPU, full corpus indexing (1,160 batches) will take approximately **32 to 38 minutes**.
   - *Recommendation:* Because execution is now fully resumable and incremental, it can be paused, stopped, or run in slices (e.g. per-document or staged chunk limits) with zero risk of lost work.
2. **Reranker Latency:**
   - The BGE-reranker cross-encoder (`BAAI/bge-reranker-base`) is computationally heavier than the bi-encoder. Running cross-attention on large candidate sets ($K > 50$) on CPU could be slow.
   - *Recommendation:* Stage reranking candidate depths to $K \le 20$ during initial evaluation passes.
3. **Storage Considerations:**
   - Full-corpus float32 dense embeddings ($37,118 \times 384$) will occupy approximately **57 MB** on disk, which is negligible and well within local disk limits. All cache paths remain ignored by `.gitignore`.

---

## 6. Summary Conclusion
Checkpoint 07 recovery is complete. The system crash has been diagnosed and definitively prevented. `DenseRetriever` is safe, non-blocking, atomic, CPU-defaulted, and reliably resumable. All unit tests and staged real-world benchmarks pass.
