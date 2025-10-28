#!/usr/bin/env python3
# main.py
"""
Index an LFW-style folder into a ChromaDB collection using InsightFace (RetinaFace + ArcFace).
Simple, clear, batch upsert to reduce overhead. Uses PersistentClient so embeddings are saved.

Usage:
    python main.py --data-dir lfw --collection lfw_faces --ctx -1 --batch 256
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import chromadb
from insightface.app import FaceAnalysis

# ---------- helpers ----------
def init_insightface(ctx_id=-1, det_size=(640, 640)):
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

def init_chroma_collection(name, persist_path="chroma_db"):
    # prefer persistent client if available
    if hasattr(chromadb, "PersistentClient"):
        client = chromadb.PersistentClient(path=persist_path)
    else:
        client = chromadb.Client()
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.create_collection(name)
    return client, collection

def is_image_file(fname):
    return fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

def gather_image_paths(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not is_image_file(fn):
                continue
            abs_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(abs_path, root)
            parts = rel.split(os.sep)
            person_label = parts[0] if len(parts) >= 2 else os.path.basename(dirpath)
            files.append((person_label, abs_path, fn))
    return files

def largest_face_item(face_list):
    if not face_list:
        return None
    best = None
    best_area = 0
    for f in face_list:
        try:
            x1, y1, x2, y2 = map(int, f.bbox[:4])
        except Exception:
            continue
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = f
    return best

def embedding_from_face(f):
    if hasattr(f, 'embedding') and f.embedding is not None:
        return np.array(f.embedding, dtype=float).flatten()
    if hasattr(f, 'normed_embedding') and f.normed_embedding is not None:
        return np.array(f.normed_embedding, dtype=float).flatten()
    return None

# ---------- main processing ----------
def process_and_upsert(app, collection, files, batch_size=256):
    total = 0
    stats = {'upserted':0, 'added':0, 'no_face':0, 'cannot_read':0, 'no_embedding':0, 'errors':0}
    batch_ids, batch_embs, batch_metas, batch_docs = [], [], [], []

    def flush_batch():
        nonlocal batch_ids, batch_embs, batch_metas, batch_docs
        if not batch_ids:
            return
        try:
            collection.upsert(ids=batch_ids, embeddings=batch_embs,
                              metadatas=batch_metas, documents=batch_docs)
        except Exception:
            try:
                collection.add(ids=batch_ids, embeddings=batch_embs,
                               metadatas=batch_metas, documents=batch_docs)
            except Exception:
                for _id, _emb, _meta, _doc in zip(batch_ids, batch_embs, batch_metas, batch_docs):
                    try:
                        collection.upsert(ids=[_id], embeddings=[_emb],
                                          metadatas=[_meta], documents=[_doc])
                    except Exception:
                        try:
                            collection.add(ids=[_id], embeddings=[_emb],
                                           metadatas=[_meta], documents=[_doc])
                        except Exception:
                            pass
        batch_ids, batch_embs, batch_metas, batch_docs = [], [], [], []

    for person, path, fname in tqdm(files, desc="Indexing"):
        total += 1
        try:
            img = cv2.imread(path)
            if img is None:
                stats['cannot_read'] += 1
                continue
            faces = app.get(img)
            if not faces:
                stats['no_face'] += 1
                continue
            f = largest_face_item(faces)
            if f is None:
                stats['no_face'] += 1
                continue
            emb = embedding_from_face(f)
            if emb is None:
                stats['no_embedding'] += 1
                continue

            uid = f"{person}/{fname}"
            meta = {"person": person, "filename": fname, "path": path}
            doc = person

            batch_ids.append(uid)
            batch_embs.append(emb.tolist())
            batch_metas.append(meta)
            batch_docs.append(doc)

            if len(batch_ids) >= batch_size:
                flush_batch()

        except Exception:
            stats['errors'] += 1
            continue

    flush_batch()

    print("Indexing finished.")
    print(f"Total files processed: {total}")
    print("Stats:", stats)

def main():
    parser = argparse.ArgumentParser(description="Index images into ChromaDB (Persistent)")
    parser.add_argument("--data-dir", default="lfw", help="root folder to index")
    parser.add_argument("--collection", default="lfw_faces", help="Chroma collection name")
    parser.add_argument("--ctx", type=int, default=-1, help="-1=CPU, 0=GPU(onnxruntime-gpu)")
    parser.add_argument("--det-size", type=int, nargs=2, default=(640,640), help="detector size WxH")
    parser.add_argument("--batch", type=int, default=256, help="batch size for upsert")
    parser.add_argument("--persist", type=str, default="chroma_db", help="persistence folder path")
    args = parser.parse_args()

    app = init_insightface(ctx_id=args.ctx, det_size=tuple(args.det_size))
    client, collection = init_chroma_collection(args.collection, persist_path=args.persist)
    files = gather_image_paths(args.data_dir)
    print(f"Found {len(files)} images under {args.data_dir}")
    process_and_upsert(app, collection, files, batch_size=args.batch)

if __name__ == "__main__":
    main()