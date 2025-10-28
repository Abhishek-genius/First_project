#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_topk.csv")

# --- Helper: extract person name ---
def extract_person_from_query(fname):
    base = fname.split("/")[-1]
    toks = [t for t in base.split("_") if not t.isdigit() and t != ""]
    if len(toks) >= 2:
        return (toks[0] + "_" + toks[1]).lower()
    return toks[0].lower() if toks else base.lower()

def extract_person_from_matchid(mid):
    if not isinstance(mid, str): 
        return None
    s = mid.strip()
    if "/" in s:
        parts = [p for p in s.split("/") if p]
        for p in parts:
            if "_" in p: 
                return p.lower()
        return parts[0].lower()
    toks = [t for t in s.split("_") if not t.isdigit()]
    if len(toks) >= 2:
        return (toks[0] + "_" + toks[1]).lower()
    return toks[0].lower() if toks else s.lower()

# --- Calculate Rank-k accuracy ---
queries = sorted(df['query_image'].unique())
Q = len(queries)

acc = {}
for k in [1, 5, 10]:
    correct = 0
    for q in queries:
        qperson = extract_person_from_query(q)
        g = df[df['query_image'] == q].sort_values('rank').head(k)
        persons = [extract_person_from_matchid(x) for x in g['match_id'].fillna('')]
        if any(p == qperson for p in persons if p):
            correct += 1
    acc[k] = correct / Q

print("Rank Accuracies:", acc)

# --- Plot ---
plt.figure(figsize=(6,4))
plt.bar([str(k) for k in acc.keys()], acc.values(), color="skyblue")
for i,(k,v) in enumerate(acc.items()):
    plt.text(i, v+0.01, f"{v:.3f}", ha="center")
plt.ylim(0,1.0)
plt.xlabel("Rank (k)")
plt.ylabel("Accuracy")
plt.title("Rank-1 / Rank-5 / Rank-10 Accuracy")
plt.tight_layout()
plt.savefig("rank_accuracy.png", dpi=150)
print("✅ Saved graph to rank_accuracy.png")
