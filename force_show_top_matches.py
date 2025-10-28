#!/usr/bin/env python3
# Force-show Top K neighbors for each detected face/variant (ignores threshold)
import os, cv2, numpy as np, json, pickle, time
from search_face import get_insightface, load_embeddings_cache, get_faiss_index, resolve_candidate_path, is_image_file

Q = "./static/uploads/46d2d736c4f846969f32c32f6b828723__s.jpeg"
COL = "lfw_faces"
CACHE_DIR = "chroma_cache"
PERSIST = "chroma_db"
TOPK = 10

app = get_insightface(ctx_id=-1)
ids, emb_np, metadatas = load_embeddings_cache(persist_path=CACHE_DIR, collection=COL, force_refresh=False)
print("Loaded:", len(ids), "embs shape:", getattr(emb_np,'shape',None))
index = get_faiss_index(emb_np, use_hnsw=True, index_path=os.path.join(CACHE_DIR,"faiss.index"))

img = cv2.imread(Q)
if img is None:
    print("QUERY IMAGE NOT FOUND:", Q); raise SystemExit(1)

variants = [(0,False),(15,False),(-15,False),(30,False),(-30,False),(90,False),(0,True)]
def rotate_image(img, angle):
    if angle%360==0: return img.copy()
    if angle==90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle==180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle==270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h,w=img.shape[:2]; M=cv2.getRotationMatrix2D((w/2,h/2), -angle,1.0); return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

for ang, flip in variants:
    v = rotate_image(img, ang)
    if flip: v = cv2.flip(v,1)
    faces = app.get(v) or []
    print(f"\n=== Variant rot{ang}{'_flip' if flip else ''}: detected {len(faces)} faces ===")
    if not faces:
        # also try whole-image fallback embedding
        emb_fb = None
        try:
            emb_fb = app.get(v)
            if emb_fb:
                # get embedding from first detection-like object if present
                f0 = emb_fb[0]
                emb0 = None
                if hasattr(f0,'embedding') and f0.embedding is not None:
                    emb0 = np.array(f0.embedding,dtype=float).flatten()
                elif hasattr(f0,'normed_embedding') and f0.normed_embedding is not None:
                    emb0 = np.array(f0.normed_embedding,dtype=float).flatten()
                if emb0 is not None:
                    q = emb0 / (np.linalg.norm(emb0) or 1.0)
                    print("Fallback whole-image embed produced; will show top-K")
                else:
                    print("No fallback embed for whole image.")
            else:
                print("No detection objects in fallback.")
        except Exception as e:
            print("Fallback detection error:", e)
    for fi, f in enumerate(faces):
        # get crop
        try:
            x1,y1,x2,y2 = map(int, f.bbox[:4])
        except Exception:
            x1,y1,x2,y2 = 0,0,v.shape[1],v.shape[0]
        sx,sy = max(0,x1), max(0,y1); sx2,sy2 = min(v.shape[1],x2), min(v.shape[0],y2)
        patch = v[sy:sy2, sx:sx2]
        # try multiple ways to get embedding
        emb = None
        try:
            if hasattr(f,'embedding') and f.embedding is not None:
                emb = np.array(f.embedding, dtype=float).flatten()
            elif hasattr(f,'normed_embedding') and f.normed_embedding is not None:
                emb = np.array(f.normed_embedding, dtype=float).flatten()
        except Exception:
            emb = None
        if emb is None:
            # run recognition on patch
            try:
                r = app.get(patch)
                if r:
                    r0 = r[0]
                    if hasattr(r0,'embedding') and r0.embedding is not None:
                        emb = np.array(r0.embedding,dtype=float).flatten()
            except Exception:
                emb = None
        if emb is None:
            print(f"face#{fi}: NO embedding extracted")
            continue
        qvec = (emb / (np.linalg.norm(emb) or 1.0)).astype('float32')
        # FAISS or dot-product
        if index is not None:
            try:
                D,I = index.search(qvec.reshape(1,-1), TOPK)
                D = D.flatten(); I = I.flatten()
                print(f"face#{fi}: FAISS top scores/ids:")
                for s,iid in zip(D,I):
                    print("  score:", float(s), "idx:", int(iid), "id:", (ids[int(iid)] if iid < len(ids) else "IDX_OOB"))
            except Exception as e:
                print("FAISS search failed:", e, "-> using dot-product fallback")
                sims = emb_np.dot(qvec)
                top = np.argsort(sims)[::-1][:TOPK]
                for t in top:
                    print("  dot:", float(sims[t]), "id:", ids[t])
        else:
            sims = emb_np.dot(qvec)
            top = np.argsort(sims)[::-1][:TOPK]
            print("face#%d: DOT top-K:" % fi)
            for t in top:
                print("  dot:", float(sims[t]), "id:", ids[t])
