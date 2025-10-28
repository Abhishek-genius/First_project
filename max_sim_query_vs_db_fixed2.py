#!/usr/bin/env python3
import os, sys, json, pickle
import numpy as np
import cv2
from search_face import get_insightface

Q = "./static/uploads/46d2d736c4f846969f32c32f6b828723__s.jpeg"
COL = "lfw_faces"

def load_from_chroma_collection():
    try:
        import chromadb
    except Exception as e:
        print("CHROMADB_NOT_AVAILABLE", e)
        return None, None, None

    if hasattr(chromadb, "PersistentClient"):
        client = chromadb.PersistentClient(path="chroma_db")
    else:
        client = chromadb.Client()

    coll = client.get_collection(COL)

    data = None
    trys = [
        {"include": ["embeddings", "metadatas", "documents"], "limit": 200000},
        {"include": ["embeddings", "metadatas"], "limit": 200000},
        {"limit": 200000},
        {}
    ]
    for params in trys:
        try:
            d = coll.get(**params)
            if d:
                data = d
                break
        except Exception as e:
            continue

    if not data:
        print("CHROMA_GET_FAILED"); return None, None, None

    embeddings = data.get("embeddings", None)
    if embeddings is None:
        emb_np = np.zeros((0,0))
    else:
        emb_np = np.array(embeddings, dtype=float)

    metadatas = data.get("metadatas", [])
    ids = data.get("ids")
    if ids is None:
        ids = data.get("documents") or []
    if ids is None or len(ids) == 0:
        ids = [str(i) for i in range(len(emb_np))]

    return emb_np, ids, metadatas

def main():
    app = get_insightface(ctx_id=-1)
    img = cv2.imread(Q)
    if img is None:
        print("IMG_NOT_FOUND"); return
    faces = app.get(img)
    if not faces:
        print("NO_FACE_DETECTED"); return
    f = faces[0]
    if hasattr(f, 'embedding') and f.embedding is not None:
        q = np.array(f.embedding, dtype=float).flatten()
    elif hasattr(f, 'normed_embedding') and f.normed_embedding is not None:
        q = np.array(f.normed_embedding, dtype=float).flatten()
    else:
        print("NO_EMBED_EXTRACTED"); return
    qn = q / (np.linalg.norm(q) or 1.0)

    embs, ids, metas = load_from_chroma_collection()
    if embs is None or embs.size == 0:
        print("DB_EMPTY_OR_LOAD_FAILED"); return

    if embs.ndim == 1: embs = embs.reshape(1,-1)
    norms = np.linalg.norm(embs, axis=1, keepdims=True); norms[norms==0]=1.0
    embs = (embs / norms).astype('float32')

    sims = embs.dot(qn)
    imax = int(sims.argmax())
    print("MAX_SIM", float(sims[imax]))
    print("MAX_IDX", imax, "ID", ids[imax] if imax < len(ids) else None)

    print("TOP5:")
    top = sims.argsort()[::-1][:5]
    for i in top:
        mid = ids[i] if i < len(ids) else None
        meta = metas[i] if metas and i < len(metas) else None
        print(f"{float(sims[i]):.6f}", mid, (meta if isinstance(meta, dict) else None))

if __name__ == "__main__":
    main()
