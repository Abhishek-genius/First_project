from flask import Flask, request, url_for, jsonify
from flask_cors import CORS
import os
import time
import logging
import traceback
from werkzeug.utils import secure_filename
import shutil
import uuid

# chromadb optional import for status endpoint and pre-checks
try:
    import chromadb
except Exception:
    chromadb = None

# base paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logfile = os.path.join(LOG_DIR, "error.log")

# configure logging early so import-time errors are captured
logging.basicConfig(level=logging.INFO, filename=logfile, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__)
CORS(app)  

# limit upload size (example: 16 MB). adjust as needed.
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MATCHED_FOLDER = os.path.join(BASE_DIR, "static", "matched")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCHED_FOLDER, exist_ok=True)

ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT


def build_url_from_path(path):
    """
    Convert an absolute path under static/ to a public URL (/static/...)
    """
    if not path:
        return None
    static_root = os.path.abspath(os.path.join(BASE_DIR, "static"))
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(static_root):
        return None
    rel = os.path.relpath(abs_path, static_root)
    return url_for('static', filename=rel.replace("\\", "/"))


# import search helper
from search_face import search_image, load_embeddings_cache  # load_embeddings_cache used in refresh endpoint

# API routes for React frontend
@app.route('/api/upload', methods=['POST'])
def api_upload():
    """
    Upload image → save to static/uploads → return URL + filename
    Avoid collisions by storing with a UUID prefix.
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"success": False, "error": "Unsupported file type"}), 400

    # avoid collisions: store as <uuid>__<origname>
    unique_name = f"{uuid.uuid4().hex}__{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    try:
        file.save(filepath)
    except Exception as e:
        logging.exception("Failed to save uploaded file")
        return jsonify({"success": False, "error": "Save failed", "exc": str(e)}), 500

    file_url = url_for('static', filename=f'uploads/{unique_name}')
    return jsonify({
        "success": True,
        "filename": unique_name,   # return the actual stored filename (use this in subsequent /api/search)
        "orig_filename": filename,
        "file_url": file_url,
        "message": "Upload successful"
    }), 200


@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Run search on uploaded file → return matches as JSON
    """
    filename = request.form.get('filename')
    if not filename:
        return jsonify({"success": False, "error": "No filename provided"}), 400

    try:
        threshold = float(request.form.get('threshold', 0.8))
    except Exception:
        threshold = 0.8

    try:
        top_k = int(request.form.get('top_k', 50))
    except Exception:
        top_k = 50

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(upload_path):
        # tolerate case where frontend sent original name but we stored with UUID prefix
        try:
            existing = sorted([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(filename)])
        except Exception:
            existing = []
        if existing:
            upload_path = os.path.join(UPLOAD_FOLDER, existing[-1])
        else:
            return jsonify({"success": False, "error": "Uploaded file not found"}), 400

    # quick pre-check: if chromadb not installed and no disk cache, return helpful error
    cache_dir = "chroma_cache"
    ids_file = os.path.join(cache_dir, "ids.json")
    emb_file = os.path.join(cache_dir, "embeddings.npy")
    if chromadb is None and (not (os.path.isfile(ids_file) and os.path.isfile(emb_file))):
        return jsonify({
            "success": False,
            "error": "Server missing chromadb and no cached embeddings found. Install chromadb or provide cache in 'chroma_cache'."
        }), 500

    try:
        t0 = time.perf_counter()
        results = search_image(
            image_path=upload_path,
            collection='lfw_faces',
            ctx=-1,
            det_size=(640, 640),
            threshold=threshold,
            top_k=top_k,
            persist='chroma_db',
            matched_output_dir=MATCHED_FOLDER,
            extra_candidate_dirs=[
                os.path.join(BASE_DIR, "static"),
                os.path.join(BASE_DIR, "static", "uploads"),
                os.path.join(BASE_DIR, "static", "images"),
                os.path.join(BASE_DIR, "dataset"),
                os.path.join(BASE_DIR, "data"),
            ],
            lfw_root_guesses=[
                os.path.join(BASE_DIR, "lfw"),
                os.path.join(BASE_DIR, "lfw-deepfunneled"),
                "/mnt/data/lfw",
                "/home/abhishek/datasets/lfw"
            ]
        )
        t1 = time.perf_counter()
        elapsed = round(t1 - t0, 3)
    except Exception as e:
        tb = traceback.format_exc()
        logging.exception("Search failed during search_image call")
        return jsonify({"success": False, "error": "Search failed", "exc": str(e), "trace": tb}), 500

    # normalize results + ensure URL
    # NOTE: search_image may return either:
    #   - a list of result dicts, or
    #   - a tuple (results_list, search_stats)
    if isinstance(results, tuple) and len(results) == 2:
        results_list, search_stats = results
    elif isinstance(results, list):
        results_list = results
        search_stats = {}
    else:
        # defensive fallback
        results_list = results or []
        search_stats = {}

    matches_for_api = []
    for r in results_list:
        url = None
        saved_path = r.get('saved_path')

        # case 1: saved_path already valid
        if saved_path and os.path.isfile(saved_path):
            url = build_url_from_path(saved_path)

        # case 2: resolve via meta
        if not url:
            meta = r.get('meta') or {}
            candidate = None
            if isinstance(meta, dict):
                candidate = (meta.get("path") or meta.get("filename") or meta.get("file") or
                             meta.get("filepath") or meta.get("document") or meta.get("doc"))

            resolved = None
            if candidate:
                tries = [
                    candidate,
                    os.path.join(BASE_DIR, candidate),
                    os.path.join(BASE_DIR, "static", candidate),
                    os.path.join(BASE_DIR, "static", "uploads", candidate),
                    os.path.join(BASE_DIR, "lfw", os.path.basename(candidate)),
                ]
                for t in tries:
                    t_abs = os.path.abspath(t)
                    if os.path.isfile(t_abs):
                        resolved = t_abs
                        break

            if resolved:
                try:
                    proj_static = os.path.abspath(os.path.join(BASE_DIR, "static"))
                    resolved_abs = os.path.abspath(resolved)
                    if resolved_abs.startswith(proj_static):
                        r["saved_path"] = resolved_abs
                        url = build_url_from_path(resolved_abs)
                    else:
                        basename = os.path.basename(resolved_abs)
                        dst_name = f"{str(r.get('id')).replace('/', '_')}__{basename}"
                        dst = os.path.join(MATCHED_FOLDER, dst_name)
                        if not os.path.isfile(dst):
                            shutil.copy2(resolved_abs, dst)
                        if os.path.isfile(dst):
                            r["saved_path"] = os.path.abspath(dst)
                            url = build_url_from_path(dst)
                except Exception as ee:
                    logging.exception("Resolve/copy failed while preparing API response")

        matches_for_api.append({
            'id': r.get('id'),
            'score': r.get('score', 0.0),
            'meta': r.get('meta'),
            'url': url,
            'variant': r.get('variant')
        })

    return jsonify({
        "success": True,
        "matches": matches_for_api,
        "elapsed": elapsed,
        "stats": search_stats
    }), 200


@app.route('/api/status', methods=['GET'])
def api_status():
    """
    Simple status endpoint to check server + Chroma collection presence and size.
    """
    status = {
        "ok": True,
        "chroma": {
            "available": False,
            "collection_exists": False,
            "n_embeddings": 0,
            "collection_name": "lfw_faces",
            "persist_path": "chroma_db"
        }
    }

    if chromadb is None:
        status["chroma"]["available"] = False
        return jsonify(status), 200

    # try to connect to persistent client if possible
    try:
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=status["chroma"]["persist_path"])
        else:
            client = chromadb.Client()
        status["chroma"]["available"] = True

        try:
            coll = client.get_collection(status["chroma"]["collection_name"])
            status["chroma"]["collection_exists"] = True
            # fetch a small sample to infer size (many chroma versions don't expose direct count)
            try:
                data = coll.get(include=["ids"], limit=1)
                ids = data.get("ids", []) or []
                data2 = coll.get(include=["ids"], limit=1000)
                ids2 = data2.get("ids", []) or []
                status["chroma"]["n_embeddings"] = len(ids2)
            except Exception:
                try:
                    cnt = coll.count()
                    status["chroma"]["n_embeddings"] = int(cnt)
                except Exception:
                    status["chroma"]["n_embeddings"] = 0
        except Exception:
            status["chroma"]["collection_exists"] = False
    except Exception:
        status["chroma"]["available"] = False

    return jsonify(status), 200


@app.route('/api/refresh-cache', methods=['POST'])
def api_refresh_cache():
    """
    Force refresh embeddings cache from Chroma (if chromadb installed).
    Use this if you update the Chroma DB and want server to reload embeddings.
    """
    if chromadb is None:
        return jsonify({"success": False, "error": "chromadb not installed on server"}), 500
    try:
        # This will re-fetch and save cache files (best-effort)
        load_embeddings_cache(persist_path="chroma_db", collection="lfw_faces", force_refresh=True)
        return jsonify({"success": True, "message": "Cache refreshed"}), 200
    except Exception as e:
        logging.exception("Cache refresh failed")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    # logging already configured early
    app.run(host="127.0.0.1", port=5000, debug=True)
