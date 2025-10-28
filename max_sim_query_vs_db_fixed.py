#!/usr/bin/env python3
"""
Robust MAX_SIM probe — handles different chromadb versions.
Saves nothing; just prints MAX_SIM, MAX_IDX and TOP5 (sim, id, metadata snippet).
"""
import os, sys, json, pickle
import numpy as np
import cv2

from search_face import get_insightface

Q = "./static/uploads/46d2d736c4f846969f32c32f6b828723__s.jpeg"
COL = "lfw_faces"
CACHE_EMB = "chroma_cache/embeddings.npy"
CACHE_IDS = "chroma_cache/ids.json"
CACHE_META = "chroma_cache/metadatas.pkl"

def load_cache_if_exists():
    if os.path.isfile(CACHE_EMB) and os.path.isfile(CACHE_IDS):
        try:
            embs = np.load(CACHE_EMB)
            ids = json.load(open(CACHE_IDS, "r"))
            metas = None
            try:
                metas = pickle.load(open(CACHE_META, "rb"))
            except Exception:
                metas = [None] * len(ids)
            return embs, ids, metas
        except Exception as e:
            print("CACHE_LOAD_FAILED", e)
    return None, None, None

def load_from_chroma_collection():
    try:
        import chromadb
    except Exception as e:
        print("CHROMADB_NOT_AVAILABLE", e)
        return None, None, None

    try:
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path="chroma_db")
        else:
            client = chromadb.Client()
    except Exception as e:
        print("CHROMA_CLIENT_FAILED", e); return None, None, None

    try:
        coll = client.get_collection(COL)
    except Exception as e:
        print("GET_COLLECTION_FAILED", e); return None, None, None

    # try several get() variants to be compatible across chroma versions
    data = None
    trys = [
        {"include": ["embeddings", "metadatas", "documents"], "limit": 200000},
        {"include": ["embeddings", "metadatas"], "limit": 200000},
        {"limit": 200000},
        {}
    ]
    for params in trys:
        try:
            data = coll.get(**params)
            if data:
                break
        except Exception as e:
            # keep trying other patterns
            data = None
            continue

    if not data:
        # as a last resort try coll.count() then fetch in smaller chunks via documents/ids if supported
        try:
            cnt = coll.count()
        except Exception:
            cnt = None
        if not cnt:
            print("CHROMA_GET_FAILURE: unable to retrieve embeddings or count."); return None, None, None

        # try chunked fetch (best-effort)
        all_ids = []
        all_embs = []
        all_metas = []
        limit = 1000
        offset = 0
        while True:
            try:
                d = coll.get(limit=limit)
            except Exception:
                break
            ids_part = d.get("ids") or d.get("documents") or []
            embs_part = d.get("embeddings") or []
            metas_part = d.get("metadatas") or []
            if ids_part:
                all_ids.extend(ids_part)
            if len(embs_part) > 0:
                all_embs.extend(embs_part)
            if metas_part:
                all_metas.extend(metas_part)
            if not ids_part and not embs_part:
                break
            offset += len(ids_part) if ids_part else limit
            if offset >= cnt:
                break
        if len(all_embs) == 0:
            print("CHROMA_EMPTY_AFTER_CHUNK_FETCH"); return None, None, None
        embs = np.array(all_embs, dtype=float)
        ids = all_ids
        metas = all_metas if all_metas else [None] * len(ids)
        return embs, ids, metas

    # parse returned data in 'data' dict
    embeddings = data.get("embeddings") or []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids")
    # Some chroma versions put ids under 'documents' instead of 'ids'
    if not ids:
        ids = data.get("documents") or []
    # If still no ids, try to assemble synthetic ids from metadatas or indices
    if not ids:
        if metadatas and isinstance(metadatas, list):
            ids = []
            for i, m in enumerate(metadatas):
                if isinstance(m, dict):
                    # attempt common fields
                    idc = m.get("id") or m.get("filename") or m.get("file") or m.get("path") or m.get("document")
                    if idc:
                        ids.append(idc)
                    else:
                        ids.append(str(i))
                else:
                    ids.append(str(i))
        else:
            ids = [str(i) for i in range(len(embeddings))]

    # ensure embeddings array
    emb_np = np.array(embeddings, dtype=float) if len(embeddings) > 0 else np.zeros((0,0))
    metas = metadatas if metadatas else [None] * len(ids)
    return emb_np, ids, metas

def main():
    # 1) check query embedding
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

    # 2) load DB embeddings (cache preferred)
    embs, ids, metas = load_cache_if_exists()
    if embs is None:
        embs, ids, metas = load_from_chroma_collection()
    if embs is None or embs.size == 0:
        print("DB_EMPTY_OR_LOAD_FAILED"); return

    # normalize DB
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    norms = np.linalg.norm(embs, axis=1, keepdims=True); norms[norms==0]=1.0
    embs = (embs / norms).astype('float32')

    sims = embs.dot(qn)
    imax = int(sims.argmax())
    print("MAX_SIM", float(sims[imax]))
    print("MAX_IDX", imax, "ID", (ids[imax] if imax < len(ids) else None))
    print("TOP5")
    top = sims.argsort()[::-1][:5]
    for i in top:
        mid = ids[i] if i < len(ids) else None
        meta = metas[i] if metas is not None and i < len(metas) else None
        print(f"{float(sims[i]):.6f}", mid, (meta if isinstance(meta, dict) else None))

if __name__ == '__main__':
    main()
