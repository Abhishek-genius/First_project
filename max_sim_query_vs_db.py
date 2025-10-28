#!/usr/bin/env python3
import numpy as np, json, os, pickle
from search_face import get_insightface
import cv2

Q="./static/uploads/46d2d736c4f846969f32c32f6b828723__s.jpeg"
app=get_insightface(ctx_id=-1)
img=cv2.imread(Q)
if img is None:
    print("IMG_NOT_FOUND"); raise SystemExit(1)
faces=app.get(img)
if not faces:
    print("NO_FACE_DETECTED"); raise SystemExit(0)
f=faces[0]
if hasattr(f,'embedding') and f.embedding is not None:
    q=np.array(f.embedding,dtype=float).flatten()
elif hasattr(f,'normed_embedding') and f.normed_embedding is not None:
    q=np.array(f.normed_embedding,dtype=float).flatten()
else:
    print("NO_EMBED_EXTRACTED"); raise SystemExit(0)

qn = q / (np.linalg.norm(q) or 1.0)

# load cache if available
if os.path.isfile("chroma_cache/embeddings.npy") and os.path.isfile("chroma_cache/ids.json"):
    embs = np.load("chroma_cache/embeddings.npy")
    ids = json.load(open("chroma_cache/ids.json"))
    try:
        metas = pickle.load(open("chroma_cache/metadatas.pkl","rb"))
    except Exception:
        metas = [None]*len(ids)
else:
    try:
        import chromadb
        if hasattr(chromadb,"PersistentClient"):
            client = chromadb.PersistentClient(path="chroma_db")
        else:
            client = chromadb.Client()
        coll = client.get_collection("lfw_faces")
        data = coll.get(include=["embeddings","ids","metadatas"], limit=200000)
        embs = np.array(data.get("embeddings",[]), dtype=float)
        ids = data.get("ids",[])
        metas = data.get("metadatas",[]) or [None]*len(ids)
    except Exception as e:
        print("LOAD_FAILED", e); raise SystemExit(1)

if embs.size == 0:
    print("DB_EMPTY"); raise SystemExit(0)

# normalize DB rows
if embs.ndim == 1: embs = embs.reshape(1,-1)
norms = np.linalg.norm(embs, axis=1, keepdims=True); norms[norms==0]=1.0
embs = (embs / norms).astype('float32')

sims = embs.dot(qn)
imax = int(np.argmax(sims))
print("MAX_SIM", float(sims[imax]))
print("MAX_IDX", imax, "ID", (ids[imax] if imax < len(ids) else None))
print("TOP5")
top = sims.argsort()[::-1][:5]
for i in top:
    mid = ids[i] if i < len(ids) else None
    meta = metas[i] if i is not None and i < len(metas) else None
    print(f"{float(sims[i]):.6f}", mid, (meta if isinstance(meta, dict) else None))
