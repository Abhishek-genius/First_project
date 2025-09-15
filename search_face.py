#!/usr/bin/env python3
# search_face.py - improved path resolution + safer copying

import os
import argparse
import cv2
import numpy as np
import shutil
from insightface.app import FaceAnalysis

try:
    import chromadb
except Exception:
    chromadb = None


def init_insightface(ctx_id=-1, det_size=(640, 640)):
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def is_image_file(fname):
    if not isinstance(fname, str):
        return False
    return fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))


def largest_face_item(face_list):
    if not face_list:
        return None
    best, best_area = None, 0
    for f in face_list:
        try:
            x1, y1, x2, y2 = map(int, f.bbox[:4])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area, best = area, f
        except Exception:
            continue
    return best


def fetch_all_embeddings_from_chroma(collection):
    try:
        data = collection.get(include=["embeddings", "metadatas", "documents"], limit=200000)
    except Exception:
        data = collection.get(limit=200000)
    if not data:
        return [], np.zeros((0, )), []
    ids = data.get("ids", [])
    embeddings = data.get("embeddings", [])
    metadatas = data.get("metadatas", [])
    if embeddings is None or len(embeddings) == 0:
        return ids, np.zeros((0, )), metadatas
    emb_np = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_np = emb_np / norms
    return ids, emb_np, metadatas


def try_import_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None


def build_faiss_index(emb_np):
    faiss = try_import_faiss()
    if faiss is None or emb_np.size == 0:
        return None
    d = emb_np.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb_np.astype("float32"))
    return index


def make_chroma_client(persist_path=None):
    if chromadb is None:
        raise RuntimeError("chromadb not installed")
    if persist_path and hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=persist_path)
    return chromadb.Client()


def resolve_candidate_path(candidate, persist_path=None, extra_dirs=None, lfw_root_guesses=None):
    """
    Try heuristics to resolve candidate -> absolute file path.
    - candidate may be absolute, relative, basename, or 'Person/Person_0001.jpg'.
    - extra_dirs: list of directories to try prefixing candidate.
    - lfw_root_guesses: list of root dirs (e.g. /path/to/lfw) to try person/basename patterns.
    """
    if not candidate or not isinstance(candidate, str):
        return None

    # 1) absolute path
    if os.path.isabs(candidate) and os.path.isfile(candidate):
        return os.path.abspath(candidate)

    # normalized candidate
    cand_norm = candidate.replace("\\", "/").lstrip("./")

    # direct checks
    candidates_to_try = []
    candidates_to_try.append(os.path.join(os.getcwd(), cand_norm))
    candidates_to_try.append(os.path.join(os.getcwd(), "static", cand_norm))
    candidates_to_try.append(os.path.join(os.getcwd(), "static", "uploads", cand_norm))
    candidates_to_try.append(os.path.join(os.getcwd(), "static", "images", cand_norm))
    if persist_path:
        candidates_to_try.append(os.path.join(persist_path, cand_norm))
        candidates_to_try.append(os.path.join(os.path.dirname(persist_path), cand_norm))

    # extra dirs
    if extra_dirs:
        for d in extra_dirs:
            candidates_to_try.append(os.path.join(d, cand_norm))

    # If candidate is basename only, also try common dataset layouts
    base = os.path.basename(cand_norm)
    # candidate may already include folder; if not, try basename guesses
    if base and base == cand_norm:
        guesses = [
            os.path.join(os.getcwd(), "lfw", base),
            os.path.join(os.getcwd(), "dataset", base),
            os.path.join(os.getcwd(), "data", base),
            os.path.join(os.getcwd(), "static", base),
            os.path.join(os.getcwd(), "static", "uploads", base),
        ]
        for g in guesses:
            candidates_to_try.append(g)
        if lfw_root_guesses:
            for root in lfw_root_guesses:
                candidates_to_try.append(os.path.join(root, base))
                if os.path.isdir(root):
                    try:
                        for name in os.listdir(root):
                            p = os.path.join(root, name, base)
                            candidates_to_try.append(p)
                    except Exception:
                        pass

    # also try basename in any extra_dirs
    if extra_dirs:
        for d in extra_dirs:
            candidates_to_try.append(os.path.join(d, base))

    # final dedupe and check
    seen = set()
    for g in candidates_to_try:
        if not g:
            continue
        try:
            gg = os.path.abspath(g)
        except Exception:
            gg = g
        if gg in seen:
            continue
        seen.add(gg)
        if os.path.isfile(gg):
            return gg
    return None


def search_image(image_path,
                 collection='lfw_faces',
                 ctx=-1,
                 det_size=(640, 640),
                 threshold=0.8,
                 top_k=50,
                 persist=None,
                 matched_output_dir='static/matched',
                 extra_candidate_dirs=None,
                 lfw_root_guesses=None):
    """
    Search a single image against ChromaDB. Returns list of matches:
     [{'id', 'score', 'meta', 'saved_path' (copied under static/matched) or None, 'orig_path' or None}]
    """
    ret = []
    if not os.path.isfile(image_path) or not is_image_file(image_path):
        return ret

    app = init_insightface(ctx_id=ctx, det_size=det_size)

    persist_path = persist if persist else "chroma_db"
    client = make_chroma_client(persist_path)

    try:
        coll = client.get_collection(collection)
    except Exception:
        print("Chroma collection not found:", collection)
        return ret

    ids, emb_np, metadatas = fetch_all_embeddings_from_chroma(coll)
    if emb_np.size == 0:
        print("No embeddings found in ChromaDB.")
        return ret

    index = build_faiss_index(emb_np)

    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image:", image_path)
        return ret

    faces = app.get(img)
    if not faces:
        print("No face detected in image.")
        return ret
    f = largest_face_item(faces)
    if f is None:
        print("No face selected.")
        return ret

    emb = None
    if hasattr(f, "embedding") and f.embedding is not None:
        emb = np.array(f.embedding, dtype=float).flatten()
    elif hasattr(f, "normed_embedding") and f.normed_embedding is not None:
        emb = np.array(f.normed_embedding, dtype=float).flatten()
    else:
        print("No embedding available for detected face.")
        return ret

    qnorm = np.linalg.norm(emb)
    if qnorm == 0:
        print("Zero embedding.")
        return ret
    qvec = emb / qnorm

    TOP_K = max(1, int(top_k))
    if index is not None:
        q = np.array(qvec, dtype="float32").reshape(1, -1)
        D, I = index.search(q, TOP_K)
        scores, idxs = D.flatten(), I.flatten()
    else:
        sims = np.dot(emb_np, qvec)
        idxs = np.argsort(sims)[::-1][:TOP_K]
        scores = [float(sims[i]) for i in idxs]

    os.makedirs(matched_output_dir, exist_ok=True)

    for score, idx in zip(scores, idxs):
        if idx >= len(ids):
            continue
        matched_id = ids[idx]
        meta = metadatas[idx] if idx < len(metadatas) else {}
        if float(score) < float(threshold):
            continue

        # extract candidate from metadata
        candidate = None
        if isinstance(meta, dict):
            candidate = (meta.get('filename') or meta.get('file') or meta.get('path') or
                         meta.get('filepath') or meta.get('document') or meta.get('doc'))

        resolved = None
        if candidate:
            resolved = resolve_candidate_path(candidate,
                                              persist_path=persist_path,
                                              extra_dirs=extra_candidate_dirs,
                                              lfw_root_guesses=lfw_root_guesses)

        saved_path = None
        orig_path = resolved
        # if resolved path found and inside static -> use directly (no copy)
        if resolved:
            proj_static = os.path.abspath(os.path.join(os.getcwd(), "static"))
            try:
                resolved_abs = os.path.abspath(resolved)
            except Exception:
                resolved_abs = resolved
            if resolved_abs.startswith(proj_static):
                saved_path = resolved_abs
            else:
                # copy into matched_output_dir so flask can serve it
                try:
                    basename = os.path.basename(resolved_abs)
                    dst_name = f"{str(matched_id).replace('/', '_')}__{basename}"
                    dst = os.path.join(matched_output_dir, dst_name)
                    shutil.copy2(resolved_abs, dst)
                    saved_path = os.path.abspath(dst)
                except Exception as e:
                    print(f"[WARN] copy failed for {resolved_abs}: {e}")
                    saved_path = None
        else:
            # If we couldn't resolve candidate, try to guess with lfw_root_guesses by searching basename
            base = os.path.basename(candidate) if candidate else None
            if base and lfw_root_guesses:
                for root in lfw_root_guesses:
                    guess = os.path.join(root, base)
                    if os.path.isfile(guess):
                        orig_path = os.path.abspath(guess)
                        try:
                            basename = os.path.basename(orig_path)
                            dst_name = f"{str(matched_id).replace('/', '_')}__{basename}"
                            dst = os.path.join(matched_output_dir, dst_name)
                            shutil.copy2(orig_path, dst)
                            saved_path = os.path.abspath(dst)
                            break
                        except Exception as e:
                            print(f"[WARN] copy failed for {orig_path}: {e}")

        ret.append({
            'id': matched_id,
            'score': float(score),
            'meta': meta or {},
            'saved_path': saved_path,
            'orig_path': orig_path
        })

    return ret


# CLI helper
def main():
    parser = argparse.ArgumentParser(description="Search single image in ChromaDB embeddings")
    parser.add_argument("image", help="path to image")
    parser.add_argument("--collection", default="lfw_faces", help="Chroma collection name")
    parser.add_argument("--ctx", type=int, default=-1, help="-1=CPU, 0=GPU")
    parser.add_argument("--det-size", type=int, nargs=2, default=(640, 640), help="detection size")
    parser.add_argument("--threshold", type=float, default=0.50, help="cosine threshold for match (0..1)")
    parser.add_argument("--top_k", type=int, default=15, help="return top K matches")
    parser.add_argument("--persist", type=str, default=None, help="Chroma persistence folder path")
    parser.add_argument("--matched-dir", type=str, default="matched_only", help="where to copy matched files")
    args = parser.parse_args()

    results = search_image(
        image_path=args.image,
        collection=args.collection,
        ctx=args.ctx,
        det_size=tuple(args.det_size),
        threshold=args.threshold,
        top_k=args.top_k,
        persist=args.persist,
        matched_output_dir=args.matched_dir
    )
    if results:
        print(f"Found {len(results)} matches >= {args.threshold}")
        for r in results:
            print(r['id'], r['score'], r.get('meta', {}), r.get('saved_path'), r.get('orig_path'))
    else:
        print("NO MATCH above threshold")


if __name__ == "__main__":
    main()
