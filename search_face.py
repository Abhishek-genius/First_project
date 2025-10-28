#!/usr/bin/env python3
"""
search_face.py - robust hybrid face search (rotated bbox mapping + forgiving alignment)
Updated: default max_faces = 1
"""
import os
import cv2
import numpy as np
import shutil
import argparse
import json
import pickle
import time
import traceback
import signal

# optional deps
try:
    import chromadb
except Exception:
    chromadb = None

# lazy/faiss import helper
def try_import_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None

# -------------------------
# Module-level singletons / caches
# -------------------------
_APP_SINGLETON = None
_EMB_CACHE = None   # tuple: (ids, emb_np, metadatas)
_FAISS_INDEX = None
_CHROMA_CLIENT = None
_CHROMA_COLLECTION_NAME = None

# -------------------------
# InsightFace singleton
# -------------------------
def get_insightface(ctx_id=-1, det_size=(640, 640), force_reload=False):
    global _APP_SINGLETON
    if _APP_SINGLETON is not None and not force_reload:
        return _APP_SINGLETON
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError("insightface not available: " + str(e))
    app = FaceAnalysis(allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    _APP_SINGLETON = app
    return _APP_SINGLETON

# -------------------------
# Utilities
# -------------------------
def is_image_file(fname):
    return isinstance(fname, str) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

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

def make_chroma_client(persist_path=None):
    global _CHROMA_CLIENT
    if chromadb is None:
        raise RuntimeError("chromadb not installed")
    if _CHROMA_CLIENT is not None:
        return _CHROMA_CLIENT
    try:
        if persist_path and hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=persist_path)
        else:
            client = chromadb.Client()
        _CHROMA_CLIENT = client
        return client
    except Exception as e:
        raise RuntimeError("Failed to create Chroma client: " + str(e))

def fetch_all_embeddings_from_chroma(collection):
    """
    Returns (ids, emb_np, metadatas)
    - emb_np: numpy array (N, d) float, normalized rows
    """
    try:
        data = collection.get(include=["embeddings", "metadatas", "documents"], limit=200000)
    except Exception:
        try:
            data = collection.get(limit=200000)
        except Exception:
            data = None

    if not data:
        return [], np.zeros((0, 0)), []

    ids = data.get("ids", []) or []
    embeddings = data.get("embeddings")

    if embeddings is None or len(embeddings) == 0:
        return ids, np.zeros((0, 0)), data.get("metadatas", [])

    emb_np = np.array(embeddings, dtype=float)
    if emb_np.ndim == 1:
        emb_np = emb_np.reshape(1, -1)

    # normalize rows
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_np = emb_np / norms

    metadatas = data.get("metadatas", [])
    return ids, emb_np, metadatas

# -------------------------
# Embeddings cache (disk + memory)
# -------------------------
def load_embeddings_cache(persist_path="chroma_db", collection=None, force_refresh=False):
    """
    Returns (ids, emb_np, metadatas)
    - caches to disk in 'chroma_cache'
    - if Chroma collection does not exist, tries disk cache instead of raising
    """
    global _EMB_CACHE, _CHROMA_COLLECTION_NAME
    cache_dir = os.path.join("chroma_cache")
    os.makedirs(cache_dir, exist_ok=True)
    ids_file = os.path.join(cache_dir, "ids.json")
    emb_file = os.path.join(cache_dir, "embeddings.npy")
    meta_file = os.path.join(cache_dir, "metadatas.pkl")

    if _EMB_CACHE is not None and not force_refresh and collection == _CHROMA_COLLECTION_NAME:
        return _EMB_CACHE

    # try disk load first (prefer local cache for speed & resilience)
    if (not force_refresh) and os.path.isfile(ids_file) and os.path.isfile(emb_file) and os.path.isfile(meta_file):
        try:
            with open(ids_file, "r") as f:
                ids = json.load(f)
            emb_np = np.load(emb_file, allow_pickle=False)
            with open(meta_file, "rb") as f:
                metadatas = pickle.load(f)
            _EMB_CACHE = (ids, emb_np, metadatas)
            _CHROMA_COLLECTION_NAME = collection
            return _EMB_CACHE
        except Exception:
            # if disk cache corrupt, fallthrough to re-fetch from Chroma
            pass

    # fetch from chroma
    if chromadb is None:
        # no chroma installed and no disk cache -> return empty set
        return [], np.zeros((0, 0)), []

    try:
        client = make_chroma_client(persist_path)
        coll = client.get_collection(collection)
        ids, emb_np, metadatas = fetch_all_embeddings_from_chroma(coll)
    except Exception as e:
        # if Chroma collection missing, try disk cache; if no disk cache, surface readable error
        try:
            if os.path.isfile(ids_file) and os.path.isfile(emb_file) and os.path.isfile(meta_file):
                with open(ids_file, "r") as f:
                    ids = json.load(f)
                emb_np = np.load(emb_file, allow_pickle=False)
                with open(meta_file, "rb") as f:
                    metadatas = pickle.load(f)
                _EMB_CACHE = (ids, emb_np, metadatas)
                _CHROMA_COLLECTION_NAME = collection
                return _EMB_CACHE
        except Exception:
            pass
        # re-raise wrapped for caller but keep message clear
        raise RuntimeError("Failed loading embeddings from Chroma: " + str(e))

    # convert to float32
    if emb_np.size > 0:
        emb_np = emb_np.astype('float32')

    # save to disk (best-effort)
    try:
        with open(ids_file, "w") as f:
            json.dump(ids, f)
        np.save(emb_file, emb_np)
        with open(meta_file, "wb") as f:
            pickle.dump(metadatas, f)
    except Exception:
        pass

    _EMB_CACHE = (ids, emb_np, metadatas)
    _CHROMA_COLLECTION_NAME = collection
    return _EMB_CACHE

# -------------------------
# FAISS index builder / loader (singleton)
# -------------------------
def get_faiss_index(emb_np, use_hnsw=True, M=32, ef_construction=200, index_path=None, force_rebuild=False, debug=False):
    """
    Returns a faiss index. Chooses IndexFlatIP for small datasets to reduce memory.
    """
    global _FAISS_INDEX
    faiss = try_import_faiss()
    if faiss is None:
        if debug:
            print("FAISS not available, falling back to brute-force dot products.")
        return None

    if _FAISS_INDEX is not None and not force_rebuild:
        return _FAISS_INDEX

    if emb_np.size == 0:
        return None

    nvecs = emb_np.shape[0]
    d = emb_np.shape[1]
    try:
        # Heuristic: for small-to-medium collections, IndexFlatIP is simpler and uses less memory.
        if not use_hnsw or nvecs < 5000:
            idx = faiss.IndexFlatIP(d)
            idx.add(emb_np.astype('float32'))
        else:
            # HNSW can be memory heavy; give user option to disable via CLI
            idx = faiss.IndexHNSWFlat(d, int(M), faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = int(ef_construction)
            idx.add(emb_np.astype('float32'))
        if index_path:
            try:
                faiss.write_index(idx, index_path)
            except Exception:
                pass
        _FAISS_INDEX = idx
        return idx
    except Exception as e:
        if debug:
            print("FAISS build failed:", e)
        return None

# -------------------------
# Path resolver
# -------------------------
def resolve_candidate_path(candidate, persist_path=None, extra_dirs=None, lfw_root_guesses=None):
    if not candidate or not isinstance(candidate, str):
        return None

    if os.path.isabs(candidate) and os.path.isfile(candidate):
        return os.path.abspath(candidate)

    cand_norm = candidate.replace("\\", "/").lstrip("./")
    candidates_to_try = [
        os.path.join(os.getcwd(), cand_norm),
        os.path.join(os.getcwd(), "static", cand_norm),
        os.path.join(os.getcwd(), "static", "uploads", cand_norm),
        os.path.join(os.getcwd(), "static", "images", cand_norm),
    ]
    if persist_path:
        candidates_to_try.append(os.path.join(persist_path, cand_norm))
        candidates_to_try.append(os.path.join(os.path.dirname(persist_path), cand_norm))

    if extra_dirs:
        for d in extra_dirs:
            candidates_to_try.append(os.path.join(d, cand_norm))

    base = os.path.basename(cand_norm)
    if base and base == cand_norm:
        guesses = [
            os.path.join(os.getcwd(), "lfw", base),
            os.path.join(os.getcwd(), "dataset", base),
            os.path.join(os.getcwd(), "data", base),
            os.path.join(os.getcwd(), "static", base),
            os.path.join(os.getcwd(), "static", "uploads", base),
        ]
        candidates_to_try.extend(guesses)
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

    if extra_dirs:
        for d in extra_dirs:
            candidates_to_try.append(os.path.join(d, base))

    seen = set()
    for g in candidates_to_try:
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

# -------------------------
# Image helpers
# -------------------------
def rotate_image(img, angle):
    if angle % 360 == 0:
        return img.copy()
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def align_face_using_kps(img, kps, output_size=(160, 160)):
    """
    Try several alignment strategies:
      - estimateAffinePartial2D (LMEDS / RANSAC)
      - fallback to getAffineTransform using first 3 points (if present)
      - fallback to centered crop + resize
    Returns aligned image (w_out, h_out) or None
    """
    try:
        if kps is None:
            return None
        arr = np.array(kps, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        w_out, h_out = output_size
        if arr.shape[0] >= 5:
            src = arr[:5].astype(np.float32)
            dst = np.array([
                [w_out * 0.30, h_out * 0.35],
                [w_out * 0.70, h_out * 0.35],
                [w_out * 0.50, h_out * 0.55],
                [w_out * 0.35, h_out * 0.80],
                [w_out * 0.65, h_out * 0.80],
            ], dtype=np.float32)
        elif arr.shape[0] >= 3:
            src = arr[:3].astype(np.float32)
            dst = np.array([
                [w_out * 0.30, h_out * 0.35],
                [w_out * 0.70, h_out * 0.35],
                [w_out * 0.50, h_out * 0.60],
            ], dtype=np.float32)
        else:
            # fallback to centered crop
            h, w = img.shape[:2]
            side = int(min(w, h) * 0.8)
            cx, cy = w // 2, h // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            crop = img[y1:y1 + side, x1:x1 + side]
            try:
                return cv2.resize(crop, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
            except Exception:
                return None
        # try robust affine
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        # fallback to classic getAffineTransform using first 3 points (if available)
        if M is None and src.shape[0] >= 3:
            try:
                M = cv2.getAffineTransform(src[:3], dst[:3])
            except Exception:
                M = None
        if M is None:
            return None
        aligned = cv2.warpAffine(img, M, (w_out, h_out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return aligned
    except Exception:
        return None

def extract_embedding_from_face(app, face_img):
    """
    Wrapper to call insightface recognition robustly.
    Returns flattened numpy vector or None.
    """
    try:
        # try direct recognition on given face_img
        res = None
        try:
            faces = app.get(face_img)
            if faces:
                f0 = faces[0]
                if hasattr(f0, 'embedding') and f0.embedding is not None:
                    res = np.array(f0.embedding, dtype=float).flatten()
                elif hasattr(f0, 'normed_embedding') and f0.normed_embedding is not None:
                    res = np.array(f0.normed_embedding, dtype=float).flatten()
        except Exception:
            res = None

        if res is not None:
            return res

        # Try scaled variant (sometimes model needs specific size)
        try:
            small = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)
            faces2 = app.get(small)
            if faces2:
                f1 = faces2[0]
                if hasattr(f1, 'embedding') and f1.embedding is not None:
                    return np.array(f1.embedding, dtype=float).flatten()
                elif hasattr(f1, 'normed_embedding') and f1.normed_embedding is not None:
                    return np.array(f1.normed_embedding, dtype=float).flatten()
        except Exception:
            pass

        return None
    except Exception:
        return None

# -------------------------
# Helper: map points from original image -> rotated/flipped image coords
# -------------------------
def map_point_for_variant(x, y, w, h, ang, flipped):
    # x,y in original image coords (w,h)
    if ang == 0:
        nx, ny = x, y
    elif ang == 90:
        nx, ny = h - y - 1, x
    elif ang == 180:
        nx, ny = w - x - 1, h - y - 1
    elif ang == 270:
        nx, ny = y, w - x - 1
    else:
        nx, ny = x, y
    if flipped:
        # when flipped horizontally, flip across width of rotated image.
        rotate_w = h if ang in (90, 270) else w
        nx = (rotate_w - nx - 1)
    return int(round(nx)), int(round(ny))

# -------------------------
# Graceful termination helper (to avoid partial kills leaving resources weird)
# -------------------------
_killed = False
def _term_handler(signum, frame):
    global _killed
    _killed = True
    # just set flag; long loops will check and terminate gracefully
signal.signal(signal.SIGTERM, _term_handler)
signal.signal(signal.SIGINT, _term_handler)

# -------------------------
# Main search (exported)
# -------------------------
def search_image(image_path, collection='lfw_faces', ctx=-1, det_size=(640, 640),
                 threshold=0.80, top_k=50, persist=None, matched_output_dir='static/matched',
                 extra_candidate_dirs=None, lfw_root_guesses=None,
                 variants_to_try=None, flip_variants=True,
                 detect_max_size=800, hnsw_efsearch=100,
                 cache_dir="chroma_cache", faiss_index_name="faiss.index", debug=False, refresh_cache=False,
                 max_faces=1, use_hnsw=True):
    """
    Balanced search:
     - Detects ALL faces (up to max_faces) and searches each
     - Maps bbox/kps to rotated/flipped variants
     - Returns aggregated best matches
    """
    t_start = time.time()
    ret = []
    if not os.path.isfile(image_path) or not is_image_file(image_path):
        if debug:
            print("search_image: invalid image_path:", image_path)
        return ret

    # init insightface (singleton)
    try:
        app = get_insightface(ctx_id=ctx, det_size=det_size, force_reload=False)
    except Exception as e:
        if debug:
            print("InsightFace init failed:", e)
        raise

    if debug:
        print("InsightFace initialized in {:.3f}s".format(time.time() - t_start))

    # ensure output dir
    os.makedirs(matched_output_dir, exist_ok=True)

    # prepare persistence/cache paths
    persist_path = persist if persist else "chroma_db"
    cache_path = cache_dir if cache_dir else "chroma_cache"
    os.makedirs(cache_path, exist_ok=True)
    faiss_index_path = os.path.join(cache_path, faiss_index_name)

    # load embeddings (cache)
    try:
        ids, emb_np, metadatas = load_embeddings_cache(persist_path=persist_path, collection=collection, force_refresh=refresh_cache)
    except Exception as e:
        if debug:
            print("load_embeddings_cache failed:", e)
            traceback.print_exc()
        raise

    if emb_np.size == 0:
        if debug:
            print("No embeddings found in cache/Chroma.")
        return ret
    if debug:
        print("Loaded embeddings: {} vectors (shape {}) in {:.3f}s".format(len(ids), emb_np.shape, time.time() - t_start))

    # build/load faiss index
    index = get_faiss_index(emb_np, use_hnsw=use_hnsw, M=32, ef_construction=200, index_path=faiss_index_path, force_rebuild=False, debug=debug)
    if index is not None:
        try:
            if hasattr(index, 'hnsw'):
                index.hnsw.efSearch = int(hnsw_efsearch)
        except Exception:
            pass
    if debug:
        print("FAISS index ready in {:.3f}s".format(time.time() - t_start))

    # ----------------------------
    # Read image with EXIF orientation fix + robust detection on rotations/flips
    # ----------------------------
    try:
        from PIL import Image, ExifTags
        pil = Image.open(image_path)
        # apply EXIF orientation (if any)
        try:
            orient_key = None
            for k, v in ExifTags.TAGS.items():
                if v == 'Orientation':
                    orient_key = k
                    break
            exif = getattr(pil, "_getexif", lambda: None)()
            if exif and orient_key in exif:
                orientation = exif[orient_key]
                if orientation == 3:
                    pil = pil.rotate(180, expand=True)
                elif orientation == 6:
                    pil = pil.rotate(270, expand=True)
                elif orientation == 8:
                    pil = pil.rotate(90, expand=True)
        except Exception:
            # ignore EXIF issues
            pass
        # convert PIL -> OpenCV (BGR)
        orig_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # fallback to plain cv2 read if PIL not available or fails
        orig_img = cv2.imread(image_path)

    if orig_img is None:
        if debug:
            print("Could not read image:", image_path)
        return ret

    h0, w0 = orig_img.shape[:2]

    # Try initial detection on original image
    try:
        det0 = app.get(orig_img)
    except Exception:
        det0 = None

    # If no detections, attempt:
    # 1) downscaled detection
    # 2) rotated variants (0,90,180,270) and flipped horizontally for each
    if not det0:
        try:
            # downscale attempt (as before)
            scale_try = 1.0
            if max(h0, w0) > detect_max_size:
                scale_try = detect_max_size / float(max(h0, w0))
            small = cv2.resize(orig_img, (int(w0 * scale_try), int(h0 * scale_try)), interpolation=cv2.INTER_LINEAR)
            det0 = app.get(small)
        except Exception:
            det0 = None

    if not det0:
        # try all rotations and flipped variants
        tried = []
        found = False
        try:
            for ang in (0, 90, 180, 270):
                if _killed:
                    break
                # rotate original image by ang
                vimg = rotate_image(orig_img, ang)
                if vimg is None:
                    continue
                # try both non-flipped and horizontally flipped
                for flipped in (False, True):
                    if _killed:
                        break
                    key = (ang, flipped)
                    if key in tried:
                        continue
                    tried.append(key)
                    test_img = vimg.copy()
                    if flipped:
                        test_img = cv2.flip(test_img, 1)
                    det_rot = None
                    try:
                        det_rot = app.get(test_img)
                    except Exception:
                        det_rot = None
                    if det_rot:
                        # success: use this oriented image as the "original" for downstream processing
                        if debug:
                            print(f"Detection succeeded on rotated image {ang}°{' flipped' if flipped else ''}. Using this orientation.")
                        orig_img = test_img
                        h0, w0 = orig_img.shape[:2]
                        det0 = det_rot
                        found = True
                        break
                if found:
                    break
        except Exception:
            # ignore and keep det0 as None if nothing found
            det0 = det0

    if not det0:
        if debug:
            print("No faces detected in original image (including rotated/flipped attempts).")
        return ret

    # Limit to top faces by area to avoid crazy loops (prevents OOM / long runtime)
    detections = []
    for f in det0:
        try:
            x1, y1, x2, y2 = map(int, f.bbox[:4])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            detections.append((area, f))
        except Exception:
            continue
    detections.sort(key=lambda x: x[0], reverse=True)
    # keep only top max_faces
    detections = [f for (_, f) in detections[:max_faces]]
    if debug:
        print(f"Using {len(detections)} detected faces (max_faces={max_faces})")

    # variants list
    if variants_to_try is None:
        variants_to_try = [0, 90, 180, 270]
    variants = []
    for ang in variants_to_try:
        variants.append((f"rot{ang}", ang, False))
        if flip_variants:
            variants.append((f"rot{ang}_flip", ang, True))

    TOP_K = max(1, int(top_k))
    agg = {}

    # Process each detected face separately
    for face_idx, f0 in enumerate(detections):
        if _killed:
            if debug:
                print("Terminated by signal, returning partial results.")
            break

        try:
            try:
                bbox_orig = list(map(int, f0.bbox[:4]))
            except Exception:
                bbox_orig = [0, 0, w0, h0]

            kps_orig = None
            if hasattr(f0, 'kps') and f0.kps is not None:
                try:
                    kps_orig = np.array(f0.kps, dtype=float)
                except Exception:
                    kps_orig = None
            elif hasattr(f0, 'landmark') and f0.landmark is not None:
                try:
                    kps_orig = np.array(f0.landmark, dtype=float)
                except Exception:
                    kps_orig = None

            # For each variant, map bbox/kps and extract embedding
            for note, ang, flipped in variants:
                if _killed:
                    break
                try:
                    vimg = rotate_image(orig_img, ang)
                    if flipped:
                        vimg = cv2.flip(vimg, 1)
                    vh, vw = vimg.shape[:2]

                    # map original bbox to this variant's coords (map both corners)
                    x1, y1, x2, y2 = bbox_orig
                    p1 = map_point_for_variant(x1, y1, w0, h0, ang, flipped)
                    p2 = map_point_for_variant(x2, y2, w0, h0, ang, flipped)
                    vx1, vy1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    vx2, vy2 = max(p1[0], p2[0]), max(p1[1], p2[1])

                    # pad area a bit
                    side_w = max(1, vx2 - vx1)
                    side_h = max(1, vy2 - vy1)
                    pad_w = int(0.18 * side_w)
                    pad_h = int(0.18 * side_h)
                    sx = max(0, vx1 - pad_w)
                    sy = max(0, vy1 - pad_h)
                    sx2 = min(vw, vx2 + pad_w)
                    sy2 = min(vh, vy2 + pad_h)
                    if sx2 <= sx or sy2 <= sy:
                        if debug:
                            print(f"[face {face_idx} variant {note}] invalid crop, skipping.")
                        continue
                    face_patch = vimg[sy:sy2, sx:sx2]

                    # prepare keypoints mapped to variant-relative coords
                    kps_variant = None
                    if kps_orig is not None:
                        try:
                            mapped = []
                            for (px, py) in kps_orig:
                                mx, my = map_point_for_variant(px, py, w0, h0, ang, flipped)
                                mapped.append([mx - sx, my - sy])
                            kps_variant = np.array(mapped, dtype=float)
                        except Exception:
                            kps_variant = None

                    # aligned crop (try alignment; fallback to resize crop)
                    aligned = None
                    if kps_variant is not None:
                        aligned = align_face_using_kps(face_patch, kps_variant, output_size=(160, 160))
                    if aligned is None:
                        try:
                            aligned = cv2.resize(face_patch, (160, 160), interpolation=cv2.INTER_LINEAR)
                        except Exception:
                            aligned = None

                    if debug:
                        try:
                            print(f"[face {face_idx} variant {note}] face_patch {face_patch.shape} aligned={'yes' if aligned is not None else 'no'}")
                        except Exception:
                            pass

                    # ------------------ embedding extraction + fallbacks ------------------
                    emb_vec = None

                    # 1) aligned crop
                    if aligned is not None:
                        emb_vec = extract_embedding_from_face(app, aligned)

                    # 2) detect on face_patch and use embedding of best local face
                    if emb_vec is None:
                        try:
                            local_faces = app.get(face_patch)
                            local_best = largest_face_item(local_faces) if local_faces else None
                            if local_best is not None:
                                if hasattr(local_best, 'embedding') and local_best.embedding is not None:
                                    emb_vec = np.array(local_best.embedding, dtype=float).flatten()
                                elif hasattr(local_best, 'normed_embedding') and local_best.normed_embedding is not None:
                                    emb_vec = np.array(local_best.normed_embedding, dtype=float).flatten()
                        except Exception:
                            emb_vec = None

                    # 3) raw recognition on face_patch
                    if emb_vec is None:
                        try:
                            emb_try = extract_embedding_from_face(app, face_patch)
                            if emb_try is not None:
                                emb_vec = emb_try
                        except Exception:
                            emb_vec = None

                    # 4) try multi-scale on face_patch (smaller and larger)
                    if emb_vec is None:
                        try:
                            for size in (112, 128, 160):
                                try_img = cv2.resize(face_patch, (size, size), interpolation=cv2.INTER_LINEAR)
                                emb_try = extract_embedding_from_face(app, try_img)
                                if emb_try is not None:
                                    emb_vec = emb_try
                                    break
                        except Exception:
                            pass

                    # 5) final fallback: use embedding from the original detection's face object (f0)
                    if emb_vec is None:
                        try:
                            if hasattr(f0, 'embedding') and f0.embedding is not None:
                                emb_vec = np.array(f0.embedding, dtype=float).flatten()
                            elif hasattr(f0, 'normed_embedding') and f0.normed_embedding is not None:
                                emb_vec = np.array(f0.normed_embedding, dtype=float).flatten()
                        except Exception:
                            emb_vec = None

                    if emb_vec is None:
                        if debug:
                            print(f"[face {face_idx} variant {note}] no embedding after fallbacks.")
                        continue

                    qnorm = np.linalg.norm(emb_vec)
                    if qnorm == 0:
                        continue
                    qvec = (emb_vec / qnorm).astype('float32')

                    # search
                    if index is not None:
                        try:
                            D, I = index.search(qvec.reshape(1, -1), TOP_K)
                            scores, idxs = D.flatten(), I.flatten()
                        except Exception:
                            sims = np.dot(emb_np, qvec)
                            idxs = np.argsort(sims)[::-1][:TOP_K]
                            scores = [float(sims[i]) for i in idxs]
                    else:
                        sims = np.dot(emb_np, qvec)
                        idxs = np.argsort(sims)[::-1][:TOP_K]
                        scores = [float(sims[i]) for i in idxs]

                    if debug:
                        try:
                            sims_all = emb_np.dot(qvec)
                            print(f"[debug face {face_idx} {note}] SIM_RANGE min={float(np.min(sims_all)):.6f} max={float(np.max(sims_all)):.6f}")
                        except Exception:
                            pass

                    for score, idx in zip(scores, idxs):
                        if idx >= len(ids):
                            if debug:
                                print(f"[face {face_idx} variant {note}] skipping idx {idx} (oob)")
                            continue
                        if float(score) < float(threshold):
                            if debug:
                                print(f"[face {face_idx} variant {note}] candidate idx={idx} score={float(score):.4f} < threshold {threshold}")
                            continue
                        matched_id = ids[idx]
                        meta = metadatas[idx] if idx < len(metadatas) else {}

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

                        saved_path, orig_path = None, resolved
                        if resolved:
                            proj_static = os.path.abspath(os.path.join(os.getcwd(), "static"))
                            try:
                                resolved_abs = os.path.abspath(resolved)
                            except Exception:
                                resolved_abs = resolved
                            if resolved_abs.startswith(proj_static):
                                saved_path = resolved_abs
                            else:
                                try:
                                    basename = os.path.basename(resolved_abs)
                                    dst_name = f"{str(matched_id).replace('/', '_')}__{basename}"
                                    dst = os.path.join(matched_output_dir, dst_name)
                                    if not os.path.isfile(dst):
                                        shutil.copy2(resolved_abs, dst)
                                    saved_path = os.path.abspath(dst)
                                except Exception:
                                    saved_path = None

                        entry = {
                            'id': matched_id,
                            'score': float(score),
                            'meta': meta or {},
                            'saved_path': saved_path,
                            'orig_path': orig_path,
                            'variant': note,
                            'source_face_index': int(face_idx)
                        }
                        prev = agg.get(matched_id)
                        if prev is None or entry['score'] > prev['score']:
                            agg[matched_id] = entry

                except Exception:
                    if debug:
                        print(f"[face {face_idx} variant {note}] exception: {traceback.format_exc()}")
                    continue

            # small cleanup per-face to reduce memory pressure
            try:
                del f0
            except Exception:
                pass

        except Exception:
            if debug:
                print(f"[face {face_idx}] exception: {traceback.format_exc()}")
            continue

    results_list = sorted(agg.values(), key=lambda x: x['score'], reverse=True)[:TOP_K]
    ret.extend(results_list)

    if debug:
        print("Search complete in {:.3f}s".format(time.time() - t_start))
    return ret

# -------------------------
# CLI helper
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Search single image in ChromaDB embeddings (robust)")
    parser.add_argument("image", help="path to image")
    parser.add_argument("--collection", default="lfw_faces", help="Chroma collection name")
    parser.add_argument("--ctx", type=int, default=-1, help="-1=CPU, 0=GPU")
    parser.add_argument("--det-size", type=int, nargs=2, default=(640, 640), help="detection size")
    parser.add_argument("--threshold", type=float, default=0.50, help="cosine threshold for match (0..1)")
    parser.add_argument("--top_k", type=int, default=15, help="return top K matches")
    parser.add_argument("--persist", type=str, default=None, help="Chroma persistence folder path")
    parser.add_argument("--matched-dir", type=str, default="matched_only", help="where to copy matched files")
    parser.add_argument("--detect-max-size", type=int, default=800, help="downscale max dim for detection")
    parser.add_argument("--hnsw-efsearch", type=int, default=100, help="hnsw efSearch param (speed/recall)")
    parser.add_argument("--no-flip", dest="no_flip", action="store_true", help="disable flip variants")
    parser.add_argument("--cache-dir", type=str, default="chroma_cache", help="where to cache embeddings/index")
    parser.add_argument("--refresh-cache", dest="refresh_cache", action="store_true", help="force refresh embeddings cache")
    parser.add_argument("--debug", dest="debug", action="store_true", help="print timing/debug info")
    parser.add_argument("--max-faces", type=int, default=1, help="maximum detected faces to process")
    parser.add_argument("--no-hnsw", dest="no_hnsw", action="store_true", help="disable HNSW faiss index (use flat index or brute-force)")
    args = parser.parse_args()

    results = search_image(
        image_path=args.image,
        collection=args.collection,
        ctx=args.ctx,
        det_size=tuple(args.det_size),
        threshold=args.threshold,
        top_k=args.top_k,
        persist=args.persist,
        matched_output_dir=args.matched_dir,
        detect_max_size=args.detect_max_size,
        hnsw_efsearch=args.hnsw_efsearch,
        flip_variants=(not args.no_flip),
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        debug=args.debug,
        max_faces=args.max_faces,
        use_hnsw=(not args.no_hnsw)
    )
    if results:
        print(f"Found {len(results)} matches >= {args.threshold}")
        for r in results:
            print(r.get('id'), r.get('score'), r.get('meta', {}), r.get('saved_path'), r.get('orig_path'), r.get('variant'), r.get('source_face_index'))
    else:
        print("NO MATCH above threshold")

if __name__ == "__main__":
    main()
