#!/usr/bin/env python3
"""
batch_search_save.py
Run search_face.search_image on all images in 20_image_plus/ 
and save top-k results to CSV.
"""

import os
import time
import pandas as pd
from search_face import search_image

IMG_DIR = "20_image_plus"
OUT_CSV = "results_topk.csv"
TOP_K = 10
THRESHOLD = 0.5

def is_image(fname):
    return fname.lower().endswith((".jpg", ".jpeg", ".png"))

def main():
    files = sorted([f for f in os.listdir(IMG_DIR) if is_image(f)])
    results = []
    for i, fname in enumerate(files, 1):
        img_path = os.path.join(IMG_DIR, fname)
        print(f"[{i}/{len(files)}] Processing: {fname}")
        start = time.time()
        try:
            res = search_image(
                image_path=img_path,
                collection="lfw_faces",
                persist="chroma_db",
                cache_dir="chroma_cache",
                top_k=TOP_K,
                detect_max_size=640,
                debug=False
            )
        except Exception as e:
            print("Error:", e)
            res = []
        elapsed = time.time() - start

        for rank, m in enumerate(res, start=1):
            results.append({
                "query_image": fname,
                "rank": rank,
                "match_id": m.get("id"),
                "score": m.get("score"),
                "elapsed_time": round(elapsed, 3)
            })

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Done! Results saved in {OUT_CSV}")

if __name__ == "__main__":
    main()
