from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
import shutil

# Import the search helper we added in search_face.py
from search_face import search_image, resolve_candidate_path

app = Flask(__name__)

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
    If path is inside project static/ dir, return url_for('static', filename=...)
    else return None.
    """
    if not path:
        return None
    static_root = os.path.abspath(os.path.join(BASE_DIR, "static"))
    try:
        abs_path = os.path.abspath(path)
    except Exception:
        return None
    if not abs_path.startswith(static_root):
        return None
    rel = os.path.relpath(abs_path, static_root)
    return url_for('static', filename=rel.replace("\\", "/"))


@app.route('/')
def home():
    return render_template("index.html", file_url=None)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template("index.html", file_url=None, msg="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", file_url=None, msg="No file selected")
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return render_template("index.html", file_url=None, msg="Unsupported file type")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    file_url = url_for('static', filename=f'uploads/{filename}')
    # default threshold changed to 0.1 (you can change)
    return render_template("index.html", file_url=file_url, filename=filename, msg="Upload successful!", threshold=0.1, top_k=50)


@app.route('/search', methods=['POST'])
def search():
    filename = request.form.get('filename')
    if not filename:
        return render_template("index.html", file_url=None, msg="No filename provided for search")

    # parameters from the form
    try:
        threshold = float(request.form.get('threshold', 0.8))
    except ValueError:
        threshold = 0.8
    try:
        top_k = int(request.form.get('top_k', 50))
    except ValueError:
        top_k = 50

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(upload_path):
        return render_template("index.html", file_url=None, msg="Uploaded file not found. Please re-upload.")

    # call the search helper with extra dirs and LFW root guesses
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
            "/mnt/data/lfw",    # common outside path (add yours if different)
            "/home/abhishek/datasets/lfw"  # example - replace with your actual dataset root if you have it
        ]
    )

    # build matches for template
    matches_for_template = []
    for r in results:
        url = None
        saved_path = r.get('saved_path')
        # 1) if saved_path exists and is under static, build url
        if saved_path and os.path.isfile(saved_path):
            url = build_url_from_path(saved_path)
        else:
            # 2) try original path from meta
            meta = r.get('meta') or {}
            candidate = None
            if isinstance(meta, dict):
                candidate = meta.get('filename') or meta.get('file') or meta.get('path') or meta.get('filepath') or meta.get('document')
            if candidate:
                resolved = resolve_candidate_path(candidate,
                                                  persist_path='chroma_db',
                                                  extra_dirs=[
                                                      os.path.join(BASE_DIR, "static"),
                                                      os.path.join(BASE_DIR, "dataset"),
                                                      os.path.join(BASE_DIR, "data")
                                                  ],
                                                  lfw_root_guesses=[
                                                      os.path.join(BASE_DIR, "lfw"),
                                                      "/mnt/data/lfw"
                                                  ])
                if resolved and os.path.isfile(resolved):
                    # if resolved inside static, make url; else copy into matched and make url
                    maybe_url = build_url_from_path(resolved)
                    if maybe_url:
                        url = maybe_url
                    else:
                        try:
                            basename = os.path.basename(resolved)
                            dst_name = f"{str(r.get('id')).replace('/', '_')}__{basename}"
                            dst = os.path.join(MATCHED_FOLDER, dst_name)
                            if not os.path.isfile(dst):
                                shutil.copy2(resolved, dst)
                            url = build_url_from_path(dst)
                        except Exception:
                            url = None

        matches_for_template.append({
            'id': r.get('id'),
            'score': r.get('score', 0.0),
            'meta': r.get('meta'),
            'url': url
        })

    file_url = url_for('static', filename=f'uploads/{filename}')
    return render_template("index.html", file_url=file_url, filename=filename, matches=matches_for_template, searched=True, threshold=threshold, top_k=top_k)


if __name__ == '__main__':
    app.run(debug=True)
