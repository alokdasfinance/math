import os
import re
import uuid
import io
import json
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import fitz  # PyMuPDF
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash,
    send_from_directory,
)

# Optional (recommended) for stitching multi-page crops into a single PNG
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -----------------------------
# App setup
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
SAVED_DIR = BASE_DIR / "saved"              # permanent saved images live here
SAVED_DB_PATH = SAVED_DIR / "saved.json"    # persistent registry

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVED_DIR.mkdir(parents=True, exist_ok=True)
if not SAVED_DB_PATH.exists():
    SAVED_DB_PATH.write_text("[]", encoding="utf-8")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
ALLOWED_EXTENSIONS = {"pdf"}

# -----------------------------
# Env header detection
#   - START: theorem-like envs (NOT proof)
#   - END: theorem-like envs OR proof
# -----------------------------
START_WORDS = r"(theorem|lemma|proposition|corollary|claim|definition|remark|example)"
START_RE = re.compile(
    rf"^\s*{START_WORDS}\b(?:\s+(\d+(\.\d+)*|[IVXLC]+))?\s*[:.\-]?\s*.*$",
    re.IGNORECASE,
)

END_WORDS = r"(theorem|lemma|proposition|corollary|claim|definition|remark|example|proof)"
END_RE = re.compile(
    rf"^\s*{END_WORDS}\b(?:\s+(\d+(\.\d+)*|[IVXLC]+))?\s*[:.\-]?\s*.*$",
    re.IGNORECASE,
)

WHITESPACE_RE = re.compile(r"\s+")

# How many points to trim before the next header/proof so it doesn't "peek" in the crop.
END_MARGIN_PX = 10.0

# How many comma-separated words to show in the left column per snippet
MAX_KEYWORDS = 60

# Tokenization for keywords (keeps letters/digits/underscore; drops punctuation)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "by", "is", "are",
    "be", "we", "let", "then", "that", "this", "these", "those", "such", "from", "at", "it", "its",
    "if", "only", "there", "exists", "given", "where", "which", "into", "than", "also", "will",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def norm(s: str) -> str:
    s = s.replace("\u2212", "-")
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def is_env_start(text: str) -> bool:
    return START_RE.match(text) is not None


def is_env_end_marker(text: str) -> bool:
    return END_RE.match(text) is not None


# -----------------------------
# PDF line extraction (for detecting env starts + boundaries)
# -----------------------------
def extract_lines(page: fitz.Page) -> List[dict]:
    d = page.get_text("dict")
    out: List[dict] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            parts = []
            rect = None
            for span in line.get("spans", []):
                t = span.get("text", "")
                if t:
                    parts.append(t)
                bbox = span.get("bbox")
                if bbox is not None:
                    r = fitz.Rect(bbox)
                    rect = r if rect is None else rect | r

            text = norm("".join(parts))
            if text and rect is not None:
                out.append({"text": text, "rect": rect})

    out.sort(key=lambda x: (x["rect"].y0, x["rect"].x0))
    return out


# -----------------------------
# Keyword extraction from a clip (NO OCR needed)
# -----------------------------
def words_in_clip(page: fitz.Page, clip: fitz.Rect) -> List[str]:
    try:
        w = page.get_text("words", clip=clip)
    except Exception:
        return []
    w.sort(key=lambda t: (t[1], t[0]))
    return [t[4] for t in w if isinstance(t[4], str) and t[4].strip()]


def keywords_from_words(words: List[str], max_keywords: int = MAX_KEYWORDS) -> str:
    seen = set()
    out = []
    for raw in words:
        for tok in TOKEN_RE.findall(raw):
            t = tok.strip().lower()
            if not t:
                continue
            if len(t) <= 1:
                continue
            if t in STOPWORDS:
                continue
            if t.isdigit():
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= max_keywords:
                    return ", ".join(out)
    return ", ".join(out)


# -----------------------------
# Rendering helpers
# -----------------------------
def expand_rect(r: fitz.Rect, page_rect: fitz.Rect, pad: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 = max(page_rect.x0, rr.x0 - pad)
    rr.y0 = max(page_rect.y0, rr.y0 - pad)
    rr.x1 = min(page_rect.x1, rr.x1 + pad)
    rr.y1 = min(page_rect.y1, rr.y1 + pad)
    return rr


def expand_rect_no_bottom(r: fitz.Rect, page_rect: fitz.Rect, pad: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 = max(page_rect.x0, rr.x0 - pad)
    rr.y0 = max(page_rect.y0, rr.y0 - pad)
    rr.x1 = min(page_rect.x1, rr.x1 + pad)
    rr.y1 = min(page_rect.y1, rr.y1)
    return rr


def render_clip_bytes(page: fitz.Page, clip: fitz.Rect, zoom: float = 2.0) -> bytes:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return pix.tobytes("png")


def stitch_pngs_vertically(png_blobs: List[bytes]) -> Optional[bytes]:
    if not PIL_AVAILABLE or not png_blobs:
        return None
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in png_blobs]
    width = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs)
    out = Image.new("RGB", (width, total_h), (255, 255, 255))
    y = 0
    for im in imgs:
        out.paste(im, (0, y))
        y += im.height
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------
# Save DB helpers
# -----------------------------
def _load_saved_db() -> List[Dict[str, Any]]:
    try:
        return json.loads(SAVED_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_saved_db(rows: List[Dict[str, Any]]) -> None:
    SAVED_DB_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _saved_key(job_id: str, image_file: str) -> str:
    return f"{job_id}:{image_file}"


def is_saved(job_id: str, image_file: str) -> bool:
    k = _saved_key(job_id, image_file)
    db = _load_saved_db()
    return any(r.get("key") == k for r in db)


def save_item(job_id: str, keywords: str, src_image_path: Path, start_page: int) -> None:
    db = _load_saved_db()
    key = _saved_key(job_id, src_image_path.name)

    if any(r.get("key") == key for r in db):
        return

    saved_id = uuid.uuid4().hex
    dest_name = f"{saved_id}.png"
    dest_path = SAVED_DIR / dest_name

    # copy bytes (works cross-platform)
    dest_path.write_bytes(src_image_path.read_bytes())

    db.append(
        {
            "id": saved_id,
            "key": key,
            "job_id": job_id,
            "start_page": int(start_page),
            "keywords": keywords,
            "image_file": dest_name,
        }
    )
    _write_saved_db(db)


def delete_saved(saved_id: str) -> None:
    db = _load_saved_db()
    keep = []
    to_delete_file = None
    for r in db:
        if r.get("id") == saved_id:
            to_delete_file = r.get("image_file")
            continue
        keep.append(r)
    _write_saved_db(keep)
    if to_delete_file:
        p = SAVED_DIR / to_delete_file
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


# -----------------------------
# Core extraction logic
# -----------------------------
@dataclass
class EnvSnap:
    start_page: int
    header: str          # comma-separated keywords
    image_file: str      # filename under outputs/<job_id>/snaps/


def find_all_env_starts(doc: fitz.Document) -> List[Tuple[int, fitz.Rect]]:
    starts: List[Tuple[int, fitz.Rect]] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        lines = extract_lines(page)
        for ln in lines:
            if is_env_start(ln["text"]):
                starts.append((pno, ln["rect"]))
    return starts


def find_next_end_marker(doc: fitz.Document, start_page: int, start_y0: float) -> Optional[Tuple[int, float]]:
    for pno in range(start_page, doc.page_count):
        page = doc.load_page(pno)
        lines = extract_lines(page)
        for ln in lines:
            if pno == start_page and ln["rect"].y0 <= start_y0 + 0.5:
                continue
            if is_env_end_marker(ln["text"]):
                return (pno, ln["rect"].y0)
    return None


def extract_env_snaps(pdf_path: Path, out_dir: Path, zoom: float = 2.0, pad: float = 10.0) -> List[EnvSnap]:
    doc = fitz.open(str(pdf_path))
    snaps_dir = out_dir / "snaps"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    starts = find_all_env_starts(doc)
    hits: List[EnvSnap] = []

    for k, (p_start, header_rect) in enumerate(starts):
        next_marker = find_next_end_marker(doc, p_start, header_rect.y0)
        if next_marker is None:
            p_end, end_y0 = doc.page_count - 1, None
        else:
            p_end, end_y0 = next_marker

        png_parts: List[bytes] = []
        all_words: List[str] = []

        if p_start == p_end:
            page = doc.load_page(p_start)
            pr = page.rect

            y0 = header_rect.y0
            y1 = pr.y1 if end_y0 is None else max(pr.y0, end_y0 - END_MARGIN_PX)
            clip = fitz.Rect(pr.x0, y0, pr.x1, y1)

            all_words.extend(words_in_clip(page, clip))

            if end_y0 is None:
                clip = expand_rect(clip, pr, pad=pad)
            else:
                clip = expand_rect_no_bottom(clip, pr, pad=pad)

            png_parts.append(render_clip_bytes(page, clip, zoom=zoom))

        else:
            # start page
            page0 = doc.load_page(p_start)
            pr0 = page0.rect
            clip0 = fitz.Rect(pr0.x0, header_rect.y0, pr0.x1, pr0.y1)
            all_words.extend(words_in_clip(page0, clip0))
            clip0 = expand_rect(clip0, pr0, pad=pad)
            png_parts.append(render_clip_bytes(page0, clip0, zoom=zoom))

            # middle pages
            for p in range(p_start + 1, p_end):
                pm = doc.load_page(p)
                prm = pm.rect
                clipm = fitz.Rect(prm.x0, prm.y0, prm.x1, prm.y1)
                all_words.extend(words_in_clip(pm, clipm))
                clipm = expand_rect(clipm, prm, pad=pad)
                png_parts.append(render_clip_bytes(pm, clipm, zoom=zoom))

            # end page
            pageN = doc.load_page(p_end)
            prN = pageN.rect
            y_end = prN.y1 if end_y0 is None else max(prN.y0, end_y0 - END_MARGIN_PX)
            clipN = fitz.Rect(prN.x0, prN.y0, prN.x1, y_end)
            all_words.extend(words_in_clip(pageN, clipN))

            if end_y0 is None:
                clipN = expand_rect(clipN, prN, pad=pad)
            else:
                clipN = expand_rect_no_bottom(clipN, prN, pad=pad)

            png_parts.append(render_clip_bytes(pageN, clipN, zoom=zoom))

        keyword_str = keywords_from_words(all_words, max_keywords=MAX_KEYWORDS) or "(no extractable text found)"

        safe_idx = f"{k+1:05d}"
        fname = f"env_{safe_idx}_p{p_start+1:04d}.png"
        out_path = snaps_dir / fname

        stitched = stitch_pngs_vertically(png_parts)
        if stitched is not None:
            out_path.write_bytes(stitched)
        else:
            out_path.write_bytes(png_parts[0])
            for t, blob in enumerate(png_parts[1:], start=2):
                (snaps_dir / f"env_{safe_idx}_part{t}.png").write_bytes(blob)

        hits.append(EnvSnap(start_page=p_start + 1, header=keyword_str, image_file=fname))

    doc.close()
    return hits


def make_zip_of_folder(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for fn in files:
                full = Path(root) / fn
                rel = full.relative_to(folder)
                z.write(full, rel.as_posix())
    buf.seek(0)
    return buf.read()


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    # Home page can show "View Saved" button/link
    return render_template("index.html", saved_count=len(_load_saved_db()), saved_url=url_for("view_saved"))


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file provided.")
        return redirect(url_for("index"))

    f = request.files["file"]
    if f.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(f.filename):
        flash("Please upload a PDF.")
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{job_id}.pdf"
    f.save(str(upload_path))

    try:
        zoom = float(request.form.get("zoom", "2.0"))
    except ValueError:
        zoom = 2.0

    try:
        pad = float(request.form.get("pad", "10.0"))
    except ValueError:
        pad = 10.0

    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        hits = extract_env_snaps(upload_path, out_dir, zoom=zoom, pad=pad)
    except Exception as e:
        flash(f"Failed to process PDF: {e}")
        return redirect(url_for("index"))

    # Catalog JSON (used by save route to know keywords, page, etc.)
    catalog_path = out_dir / "catalog.json"
    catalog_path.write_text(json.dumps([asdict(h) for h in hits], ensure_ascii=False, indent=2), encoding="utf-8")

    # ZIP
    zip_bytes = make_zip_of_folder(out_dir)
    (OUTPUT_DIR / f"{job_id}.zip").write_bytes(zip_bytes)

    # Mark which items are already saved (so templates can show "Saved" state)
    saved_map = {h.image_file: is_saved(job_id, h.image_file) for h in hits}

    return render_template(
        "results.html",
        job_id=job_id,
        total=len(hits),
        hits=hits,
        zoom=zoom,
        pad=pad,
        saved_map=saved_map,
        saved_url=url_for("view_saved"),
    )


@app.route("/snaps/<job_id>/<filename>", methods=["GET"])
def serve_snap(job_id: str, filename: str):
    snaps_dir = OUTPUT_DIR / job_id / "snaps"
    return send_from_directory(snaps_dir, filename)


@app.route("/download/<job_id>", methods=["GET"])
def download(job_id: str):
    zip_path = OUTPUT_DIR / f"{job_id}.zip"
    if not zip_path.exists():
        flash("That ZIP download is not available.")
        return redirect(url_for("index"))
    return send_file(zip_path, as_attachment=True, download_name=f"env_snaps_{job_id}.zip")


# -----------------------------
# SAVE / VIEW SAVED
# -----------------------------
@app.route("/save/<job_id>/<path:image_file>", methods=["POST"])
def save(job_id: str, image_file: str):
    """
    Saves one extracted snippet:
      - copies its image to /saved/
      - stores {keywords, start_page, image} in saved.json
    """
    # Load catalog so we can find the keywords for this image
    catalog_path = OUTPUT_DIR / job_id / "catalog.json"
    if not catalog_path.exists():
        flash("Catalog not found for this job.")
        return redirect(url_for("index"))

    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        flash("Failed to read catalog.json.")
        return redirect(url_for("index"))

    match = None
    for row in catalog:
        if row.get("image_file") == image_file:
            match = row
            break

    if match is None:
        flash("Item not found in catalog.")
        return redirect(url_for("index"))

    src_img = OUTPUT_DIR / job_id / "snaps" / image_file
    if not src_img.exists():
        flash("Image file not found.")
        return redirect(url_for("index"))

    save_item(
        job_id=job_id,
        keywords=str(match.get("header", "")),
        src_image_path=src_img,
        start_page=int(match.get("start_page", 0)),
    )

    # Return to results page
    return redirect(url_for("results_page", job_id=job_id))

@app.route("/save_all/<job_id>", methods=["POST"])
def save_all(job_id: str):
    """
    Saves every extracted snippet for a given job_id that is not already saved.
    """
    catalog_path = OUTPUT_DIR / job_id / "catalog.json"
    if not catalog_path.exists():
        flash("Catalog not found for this job.")
        return redirect(url_for("index"))

    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        flash("Failed to read catalog.json.")
        return redirect(url_for("index"))

    snaps_dir = OUTPUT_DIR / job_id / "snaps"
    if not snaps_dir.exists():
        flash("Snaps folder not found for this job.")
        return redirect(url_for("index"))

    saved_count = 0
    skipped_count = 0

    for row in catalog:
        image_file = str(row.get("image_file", "")).strip()
        if not image_file:
            continue

        if is_saved(job_id, image_file):
            skipped_count += 1
            continue

        src_img = snaps_dir / image_file
        if not src_img.exists():
            # skip silently or count as skipped
            skipped_count += 1
            continue

        save_item(
            job_id=job_id,
            keywords=str(row.get("header", "")),
            src_image_path=src_img,
            start_page=int(row.get("start_page", 0)),
        )
        saved_count += 1

    flash(f"Saved {saved_count} items. Skipped {skipped_count} already-saved/missing items.")
    return redirect(url_for("results_page", job_id=job_id))

@app.route("/unsave/<saved_id>", methods=["POST"])
def unsave(saved_id: str):
    delete_saved(saved_id)
    return redirect(url_for("view_saved"))


@app.route("/saved", methods=["GET"])
def view_saved():
    """
    Shows all saved snippets in the same layout as results.
    We reuse results.html (same table/list), but the image URLs come from /saved_img/<file>.
    """
    db = _load_saved_db()

    # Convert saved rows into "hits"-like objects the template already expects.
    # We keep .header as the comma-separated keywords, and .image_file as saved png filename.
    class _Hit:
        def __init__(self, start_page: int, header: str, image_file: str, saved_id: str):
            self.start_page = start_page
            self.header = header
            self.image_file = image_file
            self.saved_id = saved_id

    hits = [_Hit(r.get("start_page", 0), r.get("keywords", ""), r.get("image_file", ""), r.get("id", "")) for r in db]

    # For saved view, we render a different "job_id" sentinel and provide URLs to saved images.
    # Your template can:
    #   - display images via url_for('serve_saved_img', filename=hit.image_file)
    #   - show "Unsave" button that posts to url_for('unsave', saved_id=hit.saved_id)
    return render_template(
        "saved.html" if (BASE_DIR / "templates" / "saved.html").exists() else "results.html",
        job_id="__saved__",
        total=len(hits),
        hits=hits,
        saved_url=url_for("view_saved"),
        saved_mode=True,
    )


@app.route("/saved_img/<filename>", methods=["GET"])
def serve_saved_img(filename: str):
    return send_from_directory(SAVED_DIR, filename)


@app.route("/results/<job_id>", methods=["GET"])
def results_page(job_id: str):
    """
    Reload a results page from catalog.json if user returns after saving.
    """
    catalog_path = OUTPUT_DIR / job_id / "catalog.json"
    if not catalog_path.exists():
        flash("No results found for that job.")
        return redirect(url_for("index"))

    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        flash("Failed to read catalog.json.")
        return redirect(url_for("index"))

    hits = []
    for row in catalog:
        hits.append(EnvSnap(start_page=int(row.get("start_page", 0)), header=str(row.get("header", "")), image_file=str(row.get("image_file", ""))))

    saved_map = {h.image_file: is_saved(job_id, h.image_file) for h in hits}

    return render_template(
        "results.html",
        job_id=job_id,
        total=len(hits),
        hits=hits,
        zoom=2.0,
        pad=10.0,
        saved_map=saved_map,
        saved_url=url_for("view_saved"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
