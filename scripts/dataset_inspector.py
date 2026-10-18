"""
EvasionBench Dataset Inspector & Provenance Verification
Part of the FinVeritas research audit pipeline.
Verifies row counts, schemas, label distributions, and duplicate rates.
"""

import sys
import io
import urllib.request
import pandas as pd

HF_PARQUET_URL = "https://huggingface.co/datasets/FutureMa/EvasionBench/resolve/main/evasionbench_17k_eva4b_labels_dedup.parquet"

def inspect_evasionbench(url: str = HF_PARQUET_URL) -> pd.DataFrame:
    print(f"[*] Downloading metadata/parquet from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        buffer = io.BytesIO(resp.read())
    
    df = pd.read_parquet(buffer)
    print(f"[+] Successfully loaded parquet. Shape: {df.shape}")
    print(f"[+] Columns: {df.columns.tolist()}")
    
    # Class distribution
    counts = df["eva4b_label"].value_counts()
    percentages = df["eva4b_label"].value_counts(normalize=True) * 100
    dist = pd.DataFrame({"Count": counts, "Percent": percentages})
    print("\n--- Label Distribution (eva4b_label) ---")
    print(dist.to_string())
    
    # Uniqueness & duplicates
    uids = df["uid"].nunique()
    u_q = df["question"].nunique()
    u_a = df["answer"].nunique()
    u_pairs = len(df.drop_duplicates(subset=["question", "answer"]))
    
    print("\n--- Uniqueness & Leakage Indicators ---")
    print(f"Total Rows:        {len(df)}")
    print(f"Unique UIDs:       {uids} ({uids/len(df)*100:.2f}%)")
    print(f"Unique Questions:  {u_q} ({u_q/len(df)*100:.2f}%) -> {len(df) - u_q} duplicate questions")
    print(f"Unique Answers:    {u_a} ({u_a/len(df)*100:.2f}%) -> {len(df) - u_a} duplicate answers")
    print(f"Unique QA pairs:   {u_pairs} ({u_pairs/len(df)*100:.2f}%)")
    
    # Text length metrics (characters)
    q_len = df["question"].str.len()
    a_len = df["answer"].str.len()
    print("\n--- Character Length Statistics ---")
    print(f"Question Length - Mean: {q_len.mean():.1f}, Median: {q_len.median():.1f}, Max: {q_len.max()}")
    print(f"Answer Length   - Mean: {a_len.mean():.1f}, Median: {a_len.median():.1f}, Max: {a_len.max()}")
    
    return df

if __name__ == "__main__":
    inspect_evasionbench()
