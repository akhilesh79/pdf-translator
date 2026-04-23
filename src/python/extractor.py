import io
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import torch

import pdfplumber
from PIL import Image
from pdf2image import convert_from_bytes

from .detector import detect_language
from .translator import translate_to_english

PDF_MAGIC = b"%PDF-"
DIGITAL_TEXT_MIN_CHARS = 25
OCR_BATCH_TIMEOUT_S = 600
PDF_DPI = 150       # lower than default 200 — 44% fewer pixels, same OCR quality
MAX_IMAGE_DIM = 1600  # cap longest side before Surya to bound inference time


# ---------------------------------------------------------------------------
# OCR text cleaner
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    lines = text.split("\n")
    cleaned: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            cleaned.append("")
            continue
        if len(line) <= 2 and not line.isalpha():
            continue
        alpha = sum(1 for c in line if c.isalpha())
        if len(line) > 4 and alpha / len(line) < 0.25:
            continue
        line = re.sub(r'([^\w\s])\1{2,}', r'\1\1', line)
        line = re.sub(r'\s+([,.:;!?)])', r'\1', line)
        line = re.sub(r'([(])\s+', r'\1', line)
        line = re.sub(r' {2,}', ' ', line)
        cleaned.append(line)
    result_lines: list[str] = []
    prev_blank = False
    for line in cleaned:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        result_lines.append(line)
        prev_blank = is_blank
    return "\n".join(result_lines).strip()


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
# Surya OCR — returns one string per input image
# ---------------------------------------------------------------------------

def _run_surya(surya_models: dict, pil_images: list[Image.Image]) -> list[str]:
    det_predictor = surya_models["det_predictor"]
    rec_predictor = surya_models["rec_predictor"]
    with torch.inference_mode():
        results = rec_predictor(pil_images, det_predictor=det_predictor)
    pages = []
    for page_result in results:
        lines = [
            line.text for line in page_result.text_lines
            if line.text and line.text.strip()
        ]
        pages.append("\n".join(lines))
    return pages


def ocr_images(surya_models: dict, pil_images: list[Image.Image]) -> list[str]:
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_run_surya, surya_models, pil_images)
        try:
            return future.result(timeout=OCR_BATCH_TIMEOUT_S)
        except FutureTimeout:
            print("[extractor] Surya OCR batch timed out", file=sys.stderr)
            return []
        except Exception as e:
            print(f"[extractor] Surya OCR error: {e}", file=sys.stderr)
            return []


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(raw_text: str, is_scanned: bool, page_count, lang_arg: str, nllb_model) -> dict:
    if not raw_text or not raw_text.strip():
        return {"success": False, "error": "No text could be extracted from the file."}
    cleaned_text = clean_ocr_text(raw_text)
    if not cleaned_text.strip():
        return {"success": False, "error": "Text extracted but was too noisy to process."}
    source_lang = detect_language(cleaned_text)
    translated, engine = translate_to_english(cleaned_text, source_lang, nllb_model)
    return {
        "success": True,
        "isScanned": is_scanned,
        "pageCount": page_count,
        "ocrLang": lang_arg if is_scanned else None,
        "sourceLanguage": source_lang,
        "engine": engine,
        "originalText": cleaned_text,
        "translatedText": translated,
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
    page_count = 0
    digital_texts: dict[int, str] = {}   # page index → reliable digital text
    pages_need_ocr: list[int] = []       # page indices that require Surya

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if pdfplumber_looks_reliable(text):
                    digital_texts[i] = text
                else:
                    pages_need_ocr.append(i)
    except Exception as e:
        print(f"[extractor] pdfplumber failed: {e}", file=sys.stderr)
        pages_need_ocr = list(range(max(page_count, 1)))

    # Fast path: every page has reliable digital text
    if not pages_need_ocr:
        combined = "\n\n".join(digital_texts[i] for i in range(page_count))
        return _build_result(combined, False, page_count, lang_arg, nllb_model)

    # Rasterise at low DPI — only pages that need OCR are extracted from the batch
    poppler_path = os.environ.get("POPPLER_PATH") or None
    kwargs: dict = {"dpi": PDF_DPI}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    try:
        all_pil = convert_from_bytes(file_bytes, **kwargs)
    except Exception as e:
        return {"success": False, "error": f"PDF rasterization failed (is poppler installed?): {e}"}

    page_count = page_count or len(all_pil)
    ocr_pil = [_cap_image_size(all_pil[i]) for i in pages_need_ocr if i < len(all_pil)]
    ocr_pages = ocr_images(surya_models, ocr_pil) if ocr_pil else []

    # Map each OCR result back to its original page index
    ocr_map: dict[int, str] = {
        pages_need_ocr[i]: ocr_pages[i]
        for i in range(min(len(pages_need_ocr), len(ocr_pages)))
    }

    # Assemble final text in page order, mixing digital and OCR
    parts = []
    for i in range(page_count):
        if i in digital_texts:
            parts.append(digital_texts[i])
        elif i in ocr_map and ocr_map[i].strip():
            parts.append(ocr_map[i])

    combined = "\n\n".join(p for p in parts if p.strip())
    return _build_result(combined, bool(pages_need_ocr), page_count, lang_arg, nllb_model)


def _process_image(file_bytes: bytes, lang_arg: str, surya_models: dict, nllb_model) -> dict:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        return {"success": False, "error": f"Cannot open image: {e}"}

    img = _cap_image_size(img)
    ocr_pages = ocr_images(surya_models, [img])
    ocr_text = "\n\n".join(p for p in ocr_pages if p.strip())
    return _build_result(ocr_text, True, 1, lang_arg, nllb_model)
