#!/usr/bin/env python3
"""
PDF text extraction service using pdfplumber (digital) and Tesseract (OCR).

Usage:
    python pdf_extractor.py <pdf_path> [--lang <code>]

--lang values:
    'auto'       : detect script on page 1 via Tesseract OSD and pick the lang
                   automatically. Falls back to 'eng' if OSD fails.
    'eng'        : force English.
    'eng+hin'    : multi-language OCR (Tesseract syntax).
    any code     : any Tesseract lang code, e.g. 'fra', 'chi_sim', 'hin'.

Outputs a single JSON object on stdout. Diagnostics go to stderr.
"""
import argparse
import json
import os
import re
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

DIGITAL_TEXT_MIN_CHARS = 25

# Tesseract OSD reports script names like "Devanagari", "Han", "Arabic".
# Map to the best-matching Tesseract language pack.
SCRIPT_TO_LANG = {
    "Latin":       "eng",
    "Devanagari":  "hin",
    "Arabic":      "ara",
    "Han":         "chi_sim",
    "HanS":        "chi_sim",
    "HanT":        "chi_tra",
    "Japanese":    "jpn",
    "Katakana":    "jpn",
    "Hiragana":    "jpn",
    "Hangul":      "kor",
    "Korean":      "kor",
    "Cyrillic":    "rus",
    "Greek":       "ell",
    "Hebrew":      "heb",
    "Thai":        "tha",
    "Bengali":     "ben",
    "Tamil":       "tam",
    "Telugu":      "tel",
    "Gujarati":    "guj",
    "Gurmukhi":    "pan",
    "Kannada":     "kan",
    "Malayalam":   "mal",
    "Oriya":       "ori",
    "Myanmar":     "mya",
    "Armenian":    "hye",
    "Georgian":    "kat",
    "Ethiopic":    "amh",
    "Sinhala":     "sin",
    "Khmer":       "khm",
    "Lao":         "lao",
    "Tibetan":     "bod",
}


def configure_tesseract():
    candidates = [
        os.environ.get("TESSERACT_PATH"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


def tessdata_dir():
    """Best effort to locate the tessdata folder."""
    override = os.environ.get("TESSDATA_PREFIX")
    if override and os.path.isdir(override):
        return override
    tcmd = pytesseract.pytesseract.tesseract_cmd
    if tcmd and os.path.exists(tcmd):
        candidate = os.path.join(os.path.dirname(tcmd), "tessdata")
        if os.path.isdir(candidate):
            return candidate
    return None


def installed_langs():
    d = tessdata_dir()
    if not d:
        return set()
    return {f[:-len(".traineddata")]
            for f in os.listdir(d) if f.endswith(".traineddata")}


def missing_langs(lang_spec):
    """Return list of lang codes in the spec that aren't installed. '+' joins multi-lang."""
    installed = installed_langs()
    if not installed:
        return []  # can't check — let Tesseract fail with its own error
    requested = [p for p in lang_spec.split("+") if p]
    return [p for p in requested if p not in installed]


class MissingLangPackError(Exception):
    def __init__(self, lang, script):
        super().__init__(f"Missing Tesseract language pack '{lang}' for script '{script}'")
        self.lang = lang
        self.script = script


def detect_script_lang(image, default="eng"):
    """Use Tesseract's OSD to detect script; return a matching Tesseract lang.
    Raises MissingLangPackError if the detected script's lang pack isn't installed
    and the script isn't Latin."""
    try:
        osd_raw = pytesseract.image_to_osd(image)
    except Exception as e:
        print(f"[pdf_extractor] OSD failed, falling back to '{default}': {e}", file=sys.stderr)
        return default
    m = re.search(r"Script:\s*(\S+)", osd_raw)
    if not m:
        return default
    script = m.group(1)
    lang = SCRIPT_TO_LANG.get(script, default)
    print(f"[pdf_extractor] OSD detected script '{script}' -> lang '{lang}'", file=sys.stderr)
    installed = installed_langs()
    if installed and lang not in installed:
        # Latin is special — if hypothetical "Latin" lang pack were missing we'd still
        # use eng (always present). For non-Latin scripts we MUST have the pack,
        # otherwise OCR produces garbage.
        if script == "Latin":
            return default
        raise MissingLangPackError(lang, script)
    return lang


def extract_text_pdfplumber(file_path):
    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)
        parts = []
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
        text = "\n".join(parts).strip()
        return {"text": text, "pageCount": page_count}


def pdfplumber_looks_reliable(text):
    """Heuristic: pdfplumber sometimes returns garbage for PDFs with non-Unicode fonts.
    We call it reliable if the text has a decent ratio of alphabetic chars vs symbols,
    and doesn't have too many (cid:...) markers."""
    if not text or len(text) < DIGITAL_TEXT_MIN_CHARS:
        return False
    if "(cid:" in text:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters < 20:
        return False
    # ratio of alphanumerics vs total (excluding whitespace)
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return False
    alnum = sum(1 for c in non_ws if c.isalnum())
    return alnum / len(non_ws) >= 0.55


def rasterize(file_path):
    poppler_path = os.environ.get("POPPLER_PATH") or None
    kwargs = {"poppler_path": poppler_path} if poppler_path else {}
    return convert_from_path(file_path, **kwargs)


def ocr_images(images, lang):
    parts = []
    for image in images:
        extracted = pytesseract.image_to_string(image, lang=lang)
        if extracted:
            parts.append(extracted)
    return "\n".join(parts).strip()


def extract_pdf(file_path, lang_arg):
    if not os.path.exists(file_path):
        return {"text": "", "isScanned": None, "pageCount": None,
                "success": False, "error": f"File not found: {file_path}"}

    # 1. Digital extraction
    digital_text, digital_pages = "", None
    try:
        d = extract_text_pdfplumber(file_path)
        digital_text, digital_pages = d["text"], d["pageCount"]
    except Exception as e:
        print(f"[pdf_extractor] pdfplumber failed: {e}", file=sys.stderr)

    if pdfplumber_looks_reliable(digital_text):
        return {"text": digital_text, "isScanned": False,
                "pageCount": digital_pages, "success": True, "ocrLang": None}

    # 2. OCR path: rasterize once
    try:
        images = rasterize(file_path)
    except Exception as e:
        return {"text": digital_text, "isScanned": True, "pageCount": digital_pages,
                "success": False,
                "error": f"PDF rasterization failed (is poppler installed?): {e}"}

    if not images:
        return {"text": digital_text, "isScanned": True, "pageCount": digital_pages,
                "success": False, "error": "PDF produced no pages after rasterization"}

    # 3. Resolve OCR language
    detected_script = None
    if lang_arg == "auto":
        try:
            lang = detect_script_lang(images[0], default="eng")
        except MissingLangPackError as e:
            detected_script = e.script
            # For missing packs, warn but try with 'eng' as safe fallback
            print(f"[pdf_extractor] WARNING: Detected script '{e.script}' requires pack '{e.lang}' which is missing. Falling back to 'eng'.", file=sys.stderr)
            lang = "eng"
    else:
        lang = lang_arg
        miss = missing_langs(lang)
        if miss:
            return {"text": digital_text, "isScanned": True,
                    "pageCount": digital_pages or len(images),
                    "success": False,
                    "error": (f"Missing Tesseract language pack(s): {', '.join(miss)}. "
                              f"Download {miss[0]}.traineddata from "
                              f"https://github.com/tesseract-ocr/tessdata and place it in "
                              f"the Tesseract 'tessdata' folder.")}

    # 4. Run OCR
    try:
        text = ocr_images(images, lang)
    except Exception as e:
        return {"text": digital_text, "isScanned": True,
                "pageCount": digital_pages or len(images),
                "success": False,
                "error": f"OCR with lang='{lang}' failed: {e}"}

    result = {
        "text": text,
        "isScanned": True,
        "pageCount": digital_pages or len(images),
        "success": True,
        "ocrLang": lang,
    }
    if detected_script:
        result["detectedScript"] = detected_script
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--lang", default="auto",
                        help="'auto' (default), 'eng', 'hin', 'eng+hin', etc.")
    args = parser.parse_args()

    tpath = configure_tesseract()
    if tpath is None:
        print("[pdf_extractor] WARNING: tesseract binary not found; OCR will fail",
              file=sys.stderr)

    try:
        result = extract_pdf(args.file_path, args.lang)
    except Exception as e:
        result = {"text": "", "isScanned": None, "pageCount": None,
                  "success": False, "error": f"Unexpected error: {e}"}

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
