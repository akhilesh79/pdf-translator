import io
import re
import sys
import unicodedata
import numpy as np
import pdfplumber
from PIL import Image
from pdf2image import convert_from_bytes
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from .detector import detect_language
from .translator import translate_to_english

PDF_MAGIC = b"%PDF-"
DIGITAL_TEXT_MIN_CHARS = 25
OCR_PAGE_TIMEOUT_S = 60

TESSERACT_TO_EASYOCR = {
    "eng": "en", "hin": "hi", "ara": "ar", "chi_sim": "ch_sim",
    "chi_tra": "ch_tra", "jpn": "ja", "kor": "ko", "rus": "ru",
    "fra": "fr", "deu": "de", "spa": "es", "por": "pt",
}

DEFAULT_LANG_LIST = ["hi", "en"]


# ---------------------------------------------------------------------------
# OCR text cleaner — makes extracted text readable before translation
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """
    Remove OCR noise and normalize extracted text so translation produces
    coherent English sentences instead of garbled output.
    """
    # Unicode normalise (NFKC) — fixes ligatures, half-width chars, etc.
    text = unicodedata.normalize("NFKC", text)

    lines = text.split("\n")
    cleaned: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()

        # Drop empty lines at collection stage (re-added as paragraph breaks later)
        if not line:
            cleaned.append("")
            continue

        # Drop single isolated characters — almost always OCR fragments
        if len(line) <= 2 and not line.isalpha():
            continue

        # Drop lines where alphabetic ratio is below 25%
        # (scan artifacts: "~~ 0 _ _ ~~", "====" etc.)
        alpha = sum(1 for c in line if c.isalpha())
        if len(line) > 4 and alpha / len(line) < 0.25:
            continue

        # Collapse repeated punctuation (e.g. "......" → "...")
        line = re.sub(r'([^\w\s])\1{2,}', r'\1\1', line)

        # Fix spaces around punctuation
        line = re.sub(r'\s+([,.:;!?)])', r'\1', line)
        line = re.sub(r'([(])\s+', r'\1', line)

        # Collapse multiple internal spaces
        line = re.sub(r' {2,}', ' ', line)

        cleaned.append(line)

    # Collapse runs of more than one blank line into a single paragraph break
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
# PDF validation helpers
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


def resolve_lang_list(lang_arg: str) -> list[str]:
    if lang_arg == "auto":
        return DEFAULT_LANG_LIST
    raw = lang_arg.replace("+", ",").split(",")
    result = []
    for code in raw:
        code = code.strip()
        mapped = TESSERACT_TO_EASYOCR.get(code, code)
        if mapped:
            result.append(mapped)
    return result if result else DEFAULT_LANG_LIST


# ---------------------------------------------------------------------------
# EasyOCR helpers
# ---------------------------------------------------------------------------

def _ocr_page(reader, img_np: np.ndarray) -> str:
    results = reader.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(results)


def ocr_images(reader, images_np: list, lang_arg: str) -> str:
    parts = []
    with ThreadPoolExecutor(max_workers=1) as ex:
        for img_np in images_np:
            future = ex.submit(_ocr_page, reader, img_np)
            try:
                text = future.result(timeout=OCR_PAGE_TIMEOUT_S)
                if text:
                    parts.append(text)
            except FutureTimeout:
                print("[extractor] OCR page timed out, skipping", file=sys.stderr)
            except Exception as e:
                print(f"[extractor] OCR page error: {e}", file=sys.stderr)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Result builder — clean → detect → translate
# ---------------------------------------------------------------------------

def _build_result(raw_text: str, is_scanned: bool, page_count, lang_arg: str, nllb_model) -> dict:
    if not raw_text or not raw_text.strip():
        return {"success": False, "error": "No text could be extracted from the file."}

    # Clean OCR noise to improve translation quality
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
                 lang_arg: str, reader, nllb_model) -> dict:
    try:
        if is_pdf(file_bytes):
            return _process_pdf(file_bytes, lang_arg, reader, nllb_model)
        else:
            return _process_image(file_bytes, lang_arg, reader, nllb_model)
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


def _process_pdf(file_bytes: bytes, lang_arg: str, reader, nllb_model) -> dict:
    page_count = None
    digital_text = ""

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            parts = [p.extract_text() or "" for p in pdf.pages]
            digital_text = "\n".join(parts).strip()
    except Exception as e:
        print(f"[extractor] pdfplumber failed: {e}", file=sys.stderr)

    if pdfplumber_looks_reliable(digital_text):
        return _build_result(digital_text, False, page_count, lang_arg, nllb_model)

    # Scanned PDF — rasterise then OCR
    poppler_path = __import__("os").environ.get("POPPLER_PATH") or None
    try:
        kwargs = {"poppler_path": poppler_path} if poppler_path else {}
        pil_images = convert_from_bytes(file_bytes, **kwargs)
    except Exception as e:
        return {"success": False, "error": f"PDF rasterization failed (is poppler installed?): {e}"}

    page_count = page_count or len(pil_images)
    images_np = [np.array(img) for img in pil_images]
    del pil_images

    ocr_text = ocr_images(reader, images_np, lang_arg)
    return _build_result(ocr_text, True, page_count, lang_arg, nllb_model)


def _process_image(file_bytes: bytes, lang_arg: str, reader, nllb_model) -> dict:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        return {"success": False, "error": f"Cannot open image: {e}"}

    ocr_text = ocr_images(reader, [img_np], lang_arg)
    return _build_result(ocr_text, True, 1, lang_arg, nllb_model)
