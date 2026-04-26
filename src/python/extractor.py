import io
import os
import re
import sys
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import torch

import pdfplumber
from PIL import Image
from pdf2image import convert_from_bytes

from .detector import detect_language
from .translator import translate_to_english
from .indexing import LineIndex

import time as _time

PDF_MAGIC = b"%PDF-"
DIGITAL_TEXT_MIN_CHARS = 25
PDF_DPI = 100                 # 150 → 100. Smaller pixel input = less memory
                              # pressure during recognition on RAM-bound CPUs.
MAX_IMAGE_DIM = 1200          # 1600 → 1200; matches the lower DPI choice.
OCR_POOL_WORKERS = 3


def _t_now() -> float:
    return _time.perf_counter()


def _safe_detect_visuals(capped_all):
    try:
        from .visual import detect_all_visuals
        return detect_all_visuals(capped_all)
    except Exception as e:
        print(f"[extractor] visual detection failed: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Per-line OCR cleaning (replaces the old whole-text clean_ocr_text).
# Applied line-by-line so that bbox metadata stays aligned with cleaned text.
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,30}>")

def _clean_line(text: str) -> str | None:
    text = unicodedata.normalize("NFKC", text).strip()
    if not text:
        return None
    # Strip embedded HTML tags (<b>, </b>, <i>, etc.) that some PDFs leak
    # into pdfplumber's text extraction.
    text = _HTML_TAG_RE.sub("", text).strip()
    if not text:
        return None
    if len(text) <= 2 and not text.isalpha():
        return None
    alpha = sum(1 for c in text if c.isalpha())
    if len(text) > 4 and alpha / len(text) < 0.25:
        return None
    text = re.sub(r'([^\w\s])\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+([,.:;!?)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


# Back-compat shim for any external caller (the old public function).
def clean_ocr_text(text: str) -> str:
    out: list[str] = []
    prev_blank = False
    for raw in text.split("\n"):
        cleaned = _clean_line(raw)
        if cleaned is None:
            if not prev_blank and out:
                out.append("")
                prev_blank = True
            continue
        out.append(cleaned)
        prev_blank = False
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# PDF / image validation helpers
# ---------------------------------------------------------------------------

def is_pdf(file_bytes: bytes) -> bool:
    return file_bytes[:5] == PDF_MAGIC


def pdfplumber_looks_reliable(text: str) -> bool:
    if not text or len(text) < DIGITAL_TEXT_MIN_CHARS:
        return False
    if "(cid:" in text:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters < 20:
        return False
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return False
    return sum(1 for c in non_ws if c.isalnum()) / len(non_ws) >= 0.55


def _cap_image_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Surya OCR — now returns per-line (text, bbox) preserving provenance
# ---------------------------------------------------------------------------

def _run_surya(
    surya_models: dict, pil_images: list[Image.Image]
) -> list[list[tuple[str, list[float]]]]:
    det_predictor = surya_models["det_predictor"]
    rec_predictor = surya_models["rec_predictor"]
    with torch.inference_mode():
        results = rec_predictor(pil_images, det_predictor=det_predictor)
    pages: list[list[tuple[str, list[float]]]] = []
    for page_result in results:
        page_lines: list[tuple[str, list[float]]] = []
        for line in page_result.text_lines:
            text = (line.text or "").strip()
            if not text:
                continue
            bbox = list(line.bbox) if getattr(line, "bbox", None) else []
            page_lines.append((text, bbox))
        pages.append(page_lines)
    return pages


_PAGE_OCR_CACHE: dict[str, list[tuple[str, list[float]]]] = {}
_PAGE_OCR_CACHE_MAX = 256       # bounded to keep memory predictable


def _image_fingerprint(img: Image.Image) -> str:
    """Cheap content hash. Uses the small thumbnail bytes — collisions on
    actual content would require pixel-identical pages, which is what we want
    for caching."""
    import hashlib, io as _io
    thumb = img.copy()
    thumb.thumbnail((256, 256))
    buf = _io.BytesIO()
    thumb.save(buf, format="PNG", optimize=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def ocr_images(
    surya_models: dict, pil_images: list[Image.Image]
) -> list[list[tuple[str, list[float]]]]:
    """Run Surya, but skip pages whose fingerprint matches a previous result.

    Cache is in-process (per uvicorn worker); reset on restart. Hits give an
    instant return, useful when the same page recurs (cover page across PDFs,
    re-uploads with a different lang arg).
    """
    if not pil_images:
        return []
    # Compute fingerprints once
    fps = [_image_fingerprint(img) for img in pil_images]

    needs_ocr_idx: list[int] = []
    needs_ocr_imgs: list[Image.Image] = []
    cached_results: dict[int, list[tuple[str, list[float]]]] = {}
    for i, fp in enumerate(fps):
        cached = _PAGE_OCR_CACHE.get(fp)
        if cached is not None:
            cached_results[i] = cached
        else:
            needs_ocr_idx.append(i)
            needs_ocr_imgs.append(pil_images[i])

    if cached_results:
        print(f"[extractor] page-ocr cache: {len(cached_results)}/{len(pil_images)} hits",
              file=sys.stderr)

    new_results: list[list[tuple[str, list[float]]]] = []
    if needs_ocr_imgs:
        try:
            new_results = _run_surya(surya_models, needs_ocr_imgs)
        except Exception as e:
            print(f"[extractor] Surya OCR error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            new_results = [[] for _ in needs_ocr_imgs]
        # Store in cache (bounded eviction)
        for idx, fp, result in zip(needs_ocr_idx, [fps[i] for i in needs_ocr_idx], new_results):
            if len(_PAGE_OCR_CACHE) >= _PAGE_OCR_CACHE_MAX:
                _PAGE_OCR_CACHE.pop(next(iter(_PAGE_OCR_CACHE)))
            _PAGE_OCR_CACHE[fp] = result

    # Reassemble in original order
    out: list[list[tuple[str, list[float]]]] = []
    new_iter = iter(new_results)
    for i in range(len(pil_images)):
        if i in cached_results:
            out.append(cached_results[i])
        else:
            out.append(next(new_iter))
    return out


# ---------------------------------------------------------------------------
# Assemble a LineIndex from mixed digital + OCR per-page data
# ---------------------------------------------------------------------------

def _digital_text_to_lines(
    page_num_1based: int, text: str
) -> list[tuple[int, list[float], str]]:
    """Digital pdfplumber text has no per-line bbox — use empty bbox
    (provenance resolver will emit [0,0,0,0] as a sentinel meaning 'page-level')."""
    lines: list[tuple[int, list[float], str]] = []
    for raw in text.split("\n"):
        cleaned = _clean_line(raw)
        if cleaned:
            lines.append((page_num_1based, [], cleaned))
    return lines


def _ocr_pagelines_to_lines(
    page_num_1based: int, page_lines: list[tuple[str, list[float]]]
) -> list[tuple[int, list[float], str]]:
    out: list[tuple[int, list[float], str]] = []
    for text, bbox in page_lines:
        cleaned = _clean_line(text)
        if cleaned:
            out.append((page_num_1based, bbox, cleaned))
    return out


# ---------------------------------------------------------------------------
# Result builder — takes structured per-page data, produces ProcessResponse dict
# ---------------------------------------------------------------------------

def _merge_fields_and_pairs(regex_fields: list, generic_pairs: list) -> list:
    """Merge regex-extracted (high-precision) fields with generic key/value pairs.

    Strategy: for each canonical key, keep the highest-confidence value. The
    regex extractors get confidence boost 1.0 for anchored matches; generic
    pairs are 0.75–0.85, so regex wins ties when both find the same key. Pairs
    that produce keys not seen by the regex pass are appended unchanged — that
    is the whole point: surface "anything labelled" the user might want.
    """
    by_key: dict[str, list] = {}
    for f in list(regex_fields) + list(generic_pairs):
        by_key.setdefault(f.key, []).append(f)

    SINGLE = {
        "patient_name", "uhid", "claim_id", "registration_id", "age", "gender",
        "dob", "admission_date", "discharge_date", "report_date", "date",
        "hospital_name", "doctor_name", "total_amount", "ward", "bed_no",
        "department", "phone", "address", "age_gender",
        "sample_id", "collection_time", "validation_time", "print_time",
        "report_status", "package_name", "diagnosis", "chief_complaint",
        "course", "investigation", "treatment", "advice",
        "personal_history", "past_history", "allergies", "condition_at_discharge",
    }

    def _normalize_for_dedup(v: str) -> str:
        # Strip trailing punctuation so "T2DM" and "T2DM." dedupe to one.
        return re.sub(r"[\s.,;:\-]+$", "", v.strip()).lower()

    final: list = []
    for key, fs in by_key.items():
        if key in SINGLE:
            final.append(max(fs, key=lambda x: (x.confidence, len(x.value))))
        else:
            # multi-value (icd_codes, pmjay_procedure_codes…) — dedupe on
            # punctuation-normalized lowercase value.
            seen: dict[str, object] = {}
            for f in fs:
                v = _normalize_for_dedup(f.value)
                if not v:
                    continue
                cur = seen.get(v)
                if cur is None or f.confidence > cur.confidence:
                    seen[v] = f
            final.extend(seen.values())
    return final


def _build_result(
    assembled_lines: list[tuple[int, list[float], str]],
    pil_images: list[Image.Image] | None,
    is_scanned: bool,
    page_count: int,
    lang_arg: str,
    nllb_model,
    digital_tables: list | None = None,
    pre_visuals: list | None = None,
    timings: dict | None = None,
) -> dict:
    import time as _t
    timings = timings or {}

    if not assembled_lines:
        return {"success": False, "error": "No text could be extracted from the file."}

    t0 = _t.perf_counter()
    line_index = LineIndex.from_cleaned_lines(assembled_lines)
    cleaned_text = line_index.text
    if not cleaned_text.strip():
        return {"success": False, "error": "Text extracted but was too noisy to process."}
    timings["index_ms"] = int((_t.perf_counter() - t0) * 1000)

    t0 = _t.perf_counter()
    source_lang = detect_language(cleaned_text)
    timings["detect_lang_ms"] = int((_t.perf_counter() - t0) * 1000)

    t0 = _t.perf_counter()
    translated, engine = translate_to_english(cleaned_text, source_lang, nllb_model)
    timings["translate_ms"] = int((_t.perf_counter() - t0) * 1000)

    fields: list = []
    tables: list = list(digital_tables) if digital_tables else []
    visual_elements: list = list(pre_visuals) if pre_visuals else []
    timeline: list = []
    decision_verdict: str | None = None
    flags: list[str] = []

    t0 = _t.perf_counter()
    try:
        from .fields import extract_fields
        from .pairs import extract_pairs
        regex_fields = extract_fields(cleaned_text, line_index, translated if engine == "opus-mt" else None)
        generic_pairs = extract_pairs(line_index)
        fields = _merge_fields_and_pairs(regex_fields, generic_pairs)
    except Exception as e:
        print(f"[extractor] field extraction failed: {e}", file=sys.stderr)
    timings["fields_ms"] = int((_t.perf_counter() - t0) * 1000)

    t0 = _t.perf_counter()
    try:
        from .timeline import build_timeline
        timeline = build_timeline(cleaned_text, line_index)
    except Exception as e:
        print(f"[extractor] timeline build failed: {e}", file=sys.stderr)
    timings["timeline_ms"] = int((_t.perf_counter() - t0) * 1000)

    # Visual detection only here if it wasn't already run in parallel with OCR.
    if not pre_visuals and pil_images:
        t0 = _t.perf_counter()
        try:
            from .visual import detect_all_visuals
            visual_elements = detect_all_visuals(pil_images)
        except Exception as e:
            print(f"[extractor] visual detection failed: {e}", file=sys.stderr)
        timings["visual_ms"] = int((_t.perf_counter() - t0) * 1000)

    t0 = _t.perf_counter()
    try:
        from .decision import decide
        decision_verdict, flags = decide(fields, tables, visual_elements, timeline)
    except Exception as e:
        print(f"[extractor] decisioning failed: {e}", file=sys.stderr)
    timings["decision_ms"] = int((_t.perf_counter() - t0) * 1000)

    print(f"[extractor] timings: {timings}", file=sys.stderr)

    return {
        "success": True,
        "isScanned": is_scanned,
        "pageCount": page_count,
        "ocrLang": lang_arg if is_scanned else None,
        "sourceLanguage": source_lang,
        "engine": engine,
        "originalText": cleaned_text,
        "translatedText": translated,
        "fields": [f.model_dump() for f in fields],
        "tables": [t.model_dump() for t in tables],
        "visual_elements": [v.model_dump() for v in visual_elements],
        "timeline": [t.model_dump() for t in timeline],
        "decision": decision_verdict,
        "flags": flags,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_file(file_bytes: bytes, filename: str, content_type: str,
                 lang_arg: str, surya_models: dict, nllb_model) -> dict:
    try:
        if is_pdf(file_bytes):
            return _process_pdf(file_bytes, lang_arg, surya_models, nllb_model)
        else:
            return _process_image(file_bytes, lang_arg, surya_models, nllb_model)
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


def _process_pdf(file_bytes: bytes, lang_arg: str, surya_models: dict, nllb_model) -> dict:
    from .tables import extract_digital_tables

    timings: dict = {}
    page_count = 0
    digital_texts: dict[int, str] = {}
    pages_need_ocr: list[int] = []
    digital_tables: list = []

    t0 = _t_now()
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if pdfplumber_looks_reliable(text):
                    digital_texts[i] = text
                else:
                    pages_need_ocr.append(i)
            digital_pages = [p for i, p in enumerate(pdf.pages) if i in digital_texts]
            if digital_pages:
                try:
                    digital_tables = extract_digital_tables(digital_pages)
                except Exception as e:
                    print(f"[extractor] digital table extraction failed: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[extractor] pdfplumber failed: {e}", file=sys.stderr)
        pages_need_ocr = list(range(max(page_count, 1)))
    timings["pdfplumber_ms"] = int((_t_now() - t0) * 1000)

    # Fast path: every page has reliable digital text — no rasterization needed.
    if not pages_need_ocr:
        assembled: list[tuple[int, list[float], str]] = []
        for i in range(page_count):
            assembled.extend(_digital_text_to_lines(i + 1, digital_texts[i]))
        return _build_result(
            assembled, None, False, page_count, lang_arg, nllb_model,
            digital_tables=digital_tables, timings=timings,
        )

    # Need OCR — rasterize the whole PDF once, then parallelize OCR + visual scan.
    poppler_path = os.environ.get("POPPLER_PATH") or None
    kwargs: dict = {"dpi": PDF_DPI}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    t0 = _t_now()
    try:
        all_pil = convert_from_bytes(file_bytes, **kwargs)
    except Exception as e:
        return {"success": False, "error": f"PDF rasterization failed (is poppler installed?): {e}"}
    timings["rasterize_ms"] = int((_t_now() - t0) * 1000)

    page_count = page_count or len(all_pil)
    t_cap = _t_now()
    capped_all = [_cap_image_size(img) for img in all_pil]
    timings["cap_ms"] = int((_t_now() - t_cap) * 1000)
    ocr_pil = [capped_all[i] for i in pages_need_ocr if i < len(capped_all)]

    # Surya OCR (the dominant CPU cost). On a 6 GB CPU box, running visual
    # detection in parallel actually hurts because cv2/pyzbar threads contend
    # with Surya for the same physical cores. Keep them sequential.
    t_ocr = _t_now()
    ocr_page_lines = ocr_images(surya_models, ocr_pil) if ocr_pil else []
    timings["ocr_ms"] = int((_t_now() - t_ocr) * 1000)

    t_vis = _t_now()
    pre_visuals = _safe_detect_visuals(capped_all)
    timings["visual_ms"] = int((_t_now() - t_vis) * 1000)

    ocr_map: dict[int, list[tuple[str, list[float]]]] = {
        pages_need_ocr[i]: ocr_page_lines[i]
        for i in range(min(len(pages_need_ocr), len(ocr_page_lines)))
    }

    is_scanned = bool(pages_need_ocr)
    assembled: list[tuple[int, list[float], str]] = []
    for i in range(page_count):
        if i in digital_texts:
            assembled.extend(_digital_text_to_lines(i + 1, digital_texts[i]))
        elif i in ocr_map and ocr_map[i]:
            assembled.extend(_ocr_pagelines_to_lines(i + 1, ocr_map[i]))

    return _build_result(
        assembled, capped_all, is_scanned, page_count, lang_arg, nllb_model,
        digital_tables=digital_tables, pre_visuals=pre_visuals, timings=timings,
    )


def _process_image(file_bytes: bytes, lang_arg: str, surya_models: dict, nllb_model) -> dict:
    timings: dict = {}
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        return {"success": False, "error": f"Cannot open image: {e}"}

    capped = _cap_image_size(img)

    t_ocr = _t_now()
    ocr_page_lines = ocr_images(surya_models, [capped])
    timings["ocr_ms"] = int((_t_now() - t_ocr) * 1000)

    t_vis = _t_now()
    pre_visuals = _safe_detect_visuals([capped])
    timings["visual_ms"] = int((_t_now() - t_vis) * 1000)

    assembled: list[tuple[int, list[float], str]] = []
    for page_lines in ocr_page_lines:
        assembled.extend(_ocr_pagelines_to_lines(1, page_lines))

    return _build_result(
        assembled, [capped], True, 1, lang_arg, nllb_model,
        pre_visuals=pre_visuals, timings=timings,
    )
