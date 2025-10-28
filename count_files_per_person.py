#!/usr/bin/env python3
"""
plot_from_results_topk_fixed_v2.py

Improved diagnostics + CMC plotting for results_topk.csv.

Reads: results_topk.csv (query_image,rank,match_id,score,elapsed_time)
Produces:
 - cmc_multi_thresholds.png   (CMC curves for several thresholds + identity-only)
 - cmc_values_multi.csv       (CMC numbers)
 - per_query_ap_from_csv.csv  (per-query AP for default threshold)
 - diagnostics.txt            (summary + first-hit distribution)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

INPUT = "results_topk.csv"
OUT_PNG = "cmc_multi_thresholds.png"
OUT_CSV = "cmc_values_multi.csv"
OUT_PERQ = "per_query_ap_from_csv.csv"
OUT_DIAG = "cmc_diagnostics.txt"

# settings
TOP_K = 20
THRESHOLDS = [0.5, 0.3, 0.0]   # 0.0 will behave like "score ignored if person matches"
DEFAULT_THRESHOLD = 0.5

# ---------------- helpers ----------------
def extract_person_from_query(fname):
    """From query filename like 'Alejandro_Toledo_Alejandro_Toledo_0001.jpg' -> 'alejandro_toledo'"""
    if not isinstance(fname, str): 
        return None
    base = os.path.basename(fname)
    toks = base.split("_")
    toks = [t for t in toks if not t.isdigit() and t != ""]
    if len(toks) >= 2:
        return (toks[0] + "_" + toks[1]).lower()
    if toks:
        return toks[0].lower()
    return base.lower()

def extract_person_from_matchid(mid):
    """From match_id like 'Alejandro_Toledo/Alejandro_Toledo_0001.jpg' -> 'alejandro_toledo'"""
    if not isinstance(mid, str):
        return None
    s = mid.strip()
    # if path-like, take the component that is likely the person folder (usually first or second)
    if "/" in s:
        parts = [p for p in s.split("/") if p]
        # prefer a component that has underscore (likely 'Firstname_Lastname')
        for p in parts:
            if "_" in p:
                return p.lower()
        # fallback to first component
        return parts[0].lower()
    # otherwise fallback to underscore heuristic
    toks = s.split("_")
    toks = [t for t in toks if not t.isdigit() and t != ""]
    if len(toks) >= 2:
        return (toks[0] + "_" + toks[1]).lower()
    if toks:
        return toks[0].lower()
    return s.lower()

def average_precision_at_k(rel_list):
    rel = np.asarray(rel_list, dtype=int)
    if rel.sum() == 0:
        return 0.0
    precisions = []
    cum = 0
    for i, r in enumerate(rel, start=1):
        if r == 1:
            cum += 1
            precisions.append(cum / float(i))
    return float(np.mean(precisions)) if precisions else 0.0

# ---------------- main ----------------
def main():
    if not os.path.exists(INPUT):
        print("Input CSV not found:", INPUT)
        return

    df = pd.read_csv(INPUT)
    required = {"query_image", "rank", "match_id", "score", "elapsed_time"}
    if not required.issubset(set(df.columns)):
        print("CSV missing required columns:", df.columns.tolist())
        return

    # group by query and sort by rank
    queries = sorted(df['query_image'].unique())
    grouped = {}
    elapsed = {}
    for q, g in df.groupby('query_image'):
        gsorted = g.sort_values('rank')
        grouped[q] = [(row['match_id'], float(row['score']), int(row['rank'])) for _, row in gsorted.iterrows()]
        elapsed[q] = float(gsorted['elapsed_time'].iloc[0]) if len(gsorted)>0 else 0.0

    Q = len(queries)
    print(f"Found {Q} queries.")

    # compute CMC for multiple thresholds + identity-only
    cmc_by_thresh = {}
    first_hit_dist_by_thresh = {}
    per_query_ap = []

    for thr in THRESHOLDS:
        correct_at_rank = np.zeros((TOP_K, Q), dtype=int)
        first_hits = []
        any_relevant_count = 0

        for qi, qname in enumerate(queries):
            qperson = extract_person_from_query(qname)
            matches = grouped[qname]
            # filter out exact self-match where gallery filename equals query filename (rare)
            q_basename = os.path.basename(qname)
            filtered = []
            for mid, score, rank in matches:
                mid_str = "" if pd.isna(mid) else str(mid)
                # skip exact self (same filename) only
                if q_basename == os.path.basename(mid_str):
                    continue
                filtered.append((mid_str, score, rank))
            filtered = filtered[:TOP_K]

            # build relevance vector depending on thr:
            rel = []
            for mid, score, rank in filtered:
                person = extract_person_from_matchid(mid) if mid else None
                # if thr == 0.0 we treat any matching person as relevant regardless of score
                if person and person == qperson:
                    relevant = 1 if (thr == 0.0 or (score is not None and score >= thr)) else 0
                else:
                    relevant = 0
                rel.append(relevant)
            # pad
            if len(rel) < TOP_K:
                rel.extend([0]*(TOP_K - len(rel)))

            if any(rel):
                any_relevant_count += 1
                # record first hit rank
                first_rank = next((i+1 for i,v in enumerate(rel) if v==1), None)
                if first_rank is not None:
                    first_hits.append(first_rank)

            # cumulative correctness
            for r in range(1, TOP_K+1):
                correct_at_rank[r-1, qi] = 1 if any(rel[:r]) else 0

        cmc_by_thresh[thr] = correct_at_rank.mean(axis=1)  # array length TOP_K
        first_hit_dist_by_thresh[thr] = Counter(first_hits)
        print(f"Threshold {thr}: queries with any relevant = {any_relevant_count}/{Q} ({any_relevant_count/Q:.3f})")

    # compute per-query AP (for DEFAULT_THRESHOLD)
    ap_rows = []
    for qname in queries:
        qperson = extract_person_from_query(qname)
        matches = grouped[qname][:TOP_K]
        q_basename = os.path.basename(qname)
        filtered = [(mid, score, rank) for (mid,score,rank) in matches if q_basename != os.path.basename(str(mid))]
        rel = []
        for mid, score, rank in filtered:
            person = extract_person_from_matchid(mid) if mid else None
            relevant = 1 if (person is not None and person == qperson and score >= DEFAULT_THRESHOLD) else 0
            rel.append(relevant)
        if len(rel) < TOP_K:
            rel.extend([0]*(TOP_K - len(rel)))
        ap = average_precision_at_k(rel)
        ap_rows.append({"query_image": qname, "AP": ap, "num_rel": int(np.sum(rel)), "elapsed_time": elapsed.get(qname, 0.0)})

    df_ap = pd.DataFrame(ap_rows)
    df_ap.to_csv(OUT_PERQ, index=False)
    print(f"Saved per-query AP -> {OUT_PERQ}")
    mAP = df_ap['AP'].mean()
    print(f"mAP @ threshold {DEFAULT_THRESHOLD} = {mAP:.6f}")

    # Save CMC numeric values into CSV (each column is threshold)
    cmc_df = pd.DataFrame({"rank": np.arange(1, TOP_K+1)})
    for thr in THRESHOLDS:
        cmc_df[f"acc_thr_{thr}"] = cmc_by_thresh[thr]
    cmc_df.to_csv(OUT_CSV, index=False)
    print(f"Saved CMC numbers -> {OUT_CSV}")

    # Plot the CMCs overlayed
    plt.figure(figsize=(9,5))
    for thr in THRESHOLDS:
        plt.plot(np.arange(1,TOP_K+1), cmc_by_thresh[thr], marker='o', label=f"thr={thr}")
    # also plot the identity-only as separate line if not already (thr=0.0 is identity-only attempt)
    plt.xlabel("Rank (k)")
    plt.ylabel("Identification Accuracy")
    plt.title(f"CMC (top-{TOP_K}) — multiple thresholds")
    plt.xticks(np.arange(1,TOP_K+1))
    plt.ylim(0,1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"Saved plot -> {OUT_PNG}")

    # Write diagnostics file summarizing first-hit distributions for each threshold
    with open(OUT_DIAG, "w") as fh:
        fh.write(f"Total queries: {Q}\n\n")
        for thr in THRESHOLDS:
            fh.write(f"=== Threshold {thr} ===\n")
            fh.write(f"Queries with any relevant: {sum(cmc_by_thresh[thr])*Q:.0f} / {Q}\n")
            fh.write("First-hit rank distribution (top 20):\n")
            cnt = first_hit_dist_by_thresh.get(thr, Counter())
            for r in range(1, TOP_K+1):
                fh.write(f" rank {r}: {cnt.get(r,0)}\n")
            fh.write("\n")
    print(f"Saved diagnostics -> {OUT_DIAG}")

if __name__ == "__main__":
    main()
