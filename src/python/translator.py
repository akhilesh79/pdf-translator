"""Hybrid translation: opus-mt primary (offline, fast), MyMemory fallback for
edge cases (unsupported languages or low-quality opus-mt output).

Routing:
    source == "en"                          → passthrough (no translation)
    source ∈ OPUS_SUPPORTED                 → opus-mt (offline)
    source ∉ OPUS_SUPPORTED + internet OK   → MyMemory (online)
    source ∉ OPUS_SUPPORTED + no internet   → passthrough (return original)

The opus-mt result is also sanity-checked; if it returns empty or a
heavy-non-Latin string for what should be English, MyMemory is tried instead.

MyMemory free tier:
    - 5,000 chars/day per IP without email
    - 50,000 chars/day with email registered (set MYMEMORY_EMAIL in .env)
    - Each request capped at 500 chars; we chunk longer text.
"""
from __future__ import annotations

import os
import re
import sys
import time
import multiprocessing
from typing import Optional

# Use a project-local cache dir (avoids Windows permission issues with ~/.cache)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
MAX_CHUNK_WORDS = 300
MAX_BATCH_SIZE = 4

# Languages opus-mt-mul-en handles well. Unlisted langs route to MyMemory.
OPUS_SUPPORTED = {
    "en", "hi", "bn", "ur", "ne", "ml", "ar", "fr", "de", "es", "pt", "it",
    "nl", "pl", "ru", "zh", "ja", "ko", "tr", "vi", "th", "sv", "da", "fi",
    "no", "cs", "ro", "hu", "uk", "el", "he", "fa", "ms", "id",
}

# Languages opus-mt won't translate well — go straight to MyMemory.
MYMEMORY_PRIMARY = {"gu", "mr", "ta", "te", "kn", "pa"}

MYMEMORY_URL = "https://api.mymemory.translated.net/get"
MYMEMORY_EMAIL = os.environ.get("MYMEMORY_EMAIL", "").strip()
MYMEMORY_TIMEOUT_S = 8
MYMEMORY_CHUNK_CHARS = 480       # < 500 to leave headroom for url-encoding


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_by_paragraph(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= MAX_CHUNK_WORDS:
            chunks.append(para)
        else:
            group: list[str] = []
            group_words = 0
            for sentence in para.replace(". ", ".\n").split("\n"):
                sw = len(sentence.split())
                if group_words + sw > MAX_CHUNK_WORDS and group:
                    chunks.append(" ".join(group))
                    group, group_words = [], 0
                group.append(sentence)
                group_words += sw
            if group:
                chunks.append(" ".join(group))
    return chunks if chunks else [text.strip()]


def _chunk_by_chars(text: str, limit: int) -> list[str]:
    """Chunk text into <=limit-char pieces, breaking on sentence/space when possible."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    cur = ""
    for sent in re.split(r"(?<=[.!?।])\s+", text):
        if len(cur) + len(sent) + 1 <= limit:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                pieces.append(cur)
            if len(sent) <= limit:
                cur = sent
            else:
                # hard-split very long sentence
                while len(sent) > limit:
                    pieces.append(sent[:limit])
                    sent = sent[limit:]
                cur = sent
    if cur:
        pieces.append(cur)
    return pieces


# ---------------------------------------------------------------------------
# Local model: opus-mt-mul-en
# ---------------------------------------------------------------------------

def load_nllb_model() -> dict:
    """Load Helsinki-NLP/opus-mt-mul-en (MarianMT, ~78M params)."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    _phys = max(1, multiprocessing.cpu_count() // 2)
    torch.set_num_threads(_phys)
    print(f"[translator] Loading {MODEL_NAME} on CPU (threads={_phys})...", file=sys.stderr)

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("[translator] Translation model ready (CPU).", file=sys.stderr)
    if MYMEMORY_EMAIL:
        print(f"[translator] MyMemory fallback enabled (email registered).", file=sys.stderr)
    else:
        print("[translator] MyMemory fallback enabled (anonymous, 5K chars/day).", file=sys.stderr)
    return {"model": model, "tokenizer": tokenizer, "device": "cpu"}


def _opus_translate_batch(chunks: list[str], model, tokenizer) -> list[str]:
    import torch
    inputs = tokenizer(
        chunks, return_tensors="pt", truncation=True, max_length=512, padding=True,
    )
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_length=512, num_beams=1, do_sample=False,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in out_ids]


def _opus_translate(text: str, bundle: dict) -> str:
    chunks = _chunk_by_paragraph(text)
    out_parts: list[str] = []
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i: i + MAX_BATCH_SIZE]
        try:
            results = _opus_translate_batch(batch, bundle["model"], bundle["tokenizer"])
            out_parts.extend(results)
        except Exception as e:
            print(f"[translator] opus batch {i} failed ({e}); per-chunk fallback", file=sys.stderr)
            for chunk in batch:
                try:
                    out_parts.extend(_opus_translate_batch([chunk], bundle["model"], bundle["tokenizer"]))
                except Exception:
                    out_parts.append(chunk)
    return "\n\n".join(out_parts)


# ---------------------------------------------------------------------------
# MyMemory client
# ---------------------------------------------------------------------------

def _mymemory_call(text: str, src_lang: str, target_lang: str = "en") -> Optional[str]:
    """Call MyMemory for a single chunk. Returns None on any failure."""
    if not text.strip():
        return ""
    try:
        import urllib.parse, urllib.request, json
    except Exception:
        return None
    params = {
        "q": text,
        "langpair": f"{src_lang}|{target_lang}",
    }
    if MYMEMORY_EMAIL:
        params["de"] = MYMEMORY_EMAIL
    url = MYMEMORY_URL + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pdf-translator/1.0"})
        with urllib.request.urlopen(req, timeout=MYMEMORY_TIMEOUT_S) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"[translator] MyMemory call failed: {e}", file=sys.stderr)
        return None
    rd = data.get("responseData") or {}
    translated = (rd.get("translatedText") or "").strip()
    status = data.get("responseStatus")
    if not translated:
        return None
    if isinstance(status, int) and status >= 400:
        # Quota exceeded etc.
        print(f"[translator] MyMemory non-OK status: {status} {data.get('responseDetails','')}",
              file=sys.stderr)
        return None
    # MyMemory often echoes input on quota — detect that.
    if translated.strip().lower() == text.strip().lower():
        return None
    return translated


def _mymemory_translate(text: str, src_lang: str) -> Optional[str]:
    """Chunked MyMemory translation. Returns None if any chunk fails."""
    chunks = _chunk_by_chars(text, MYMEMORY_CHUNK_CHARS)
    out: list[str] = []
    for c in chunks:
        translated = _mymemory_call(c, src_lang)
        if translated is None:
            return None
        out.append(translated)
        # Be polite — small delay between calls.
        if len(chunks) > 1:
            time.sleep(0.05)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Quality sniff for opus-mt output
# ---------------------------------------------------------------------------

def _opus_output_looks_bad(translated: str, source_text: str) -> bool:
    """Heuristic: did opus-mt fail silently?"""
    t = translated.strip()
    if not t:
        return True
    if len(t) < min(10, len(source_text) // 10):
        return True
    # Translated should be mostly Latin alphabet (we forced English target).
    latin = sum(1 for c in t if c.isascii() and c.isalpha())
    if len(t) > 20 and latin / max(1, len(t)) < 0.40:
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry point — hybrid
# ---------------------------------------------------------------------------

def translate_to_english(text: str, source_lang: str, nllb_bundle: dict) -> tuple[str, str]:
    """Returns (translated_text, engine_id)."""
    if not text or not text.strip():
        return "", "noop"

    if source_lang == "en":
        return text, "passthrough"

    # 1. MyMemory primary for languages opus-mt is known to handle poorly.
    if source_lang in MYMEMORY_PRIMARY:
        mm = _mymemory_translate(text, source_lang)
        if mm:
            return mm, "mymemory"
        # MyMemory failed (quota / no internet) — try opus-mt anyway.
        try:
            opus = _opus_translate(text, nllb_bundle)
            if not _opus_output_looks_bad(opus, text):
                return opus, "opus-mt-fallback"
        except Exception as e:
            print(f"[translator] opus-mt fallback failed: {e}", file=sys.stderr)
        return text, "passthrough-failed"

    # 2. Opus-mt primary for supported (and unknown) source languages.
    print(f"[translator] opus-mt: source={source_lang}, chars={len(text)}", file=sys.stderr)
    try:
        opus = _opus_translate(text, nllb_bundle)
    except Exception as e:
        print(f"[translator] opus-mt failed: {e}", file=sys.stderr)
        opus = ""

    if not _opus_output_looks_bad(opus, text):
        return opus, "opus-mt"

    # 3. Opus output looks bad — escalate to MyMemory if the network is reachable.
    print(f"[translator] opus output looks bad; trying MyMemory fallback...", file=sys.stderr)
    mm = _mymemory_translate(text, source_lang)
    if mm:
        return mm, "mymemory-fallback"

    # 4. Last resort — return whatever opus produced (may be empty/garbage)
    #    or the original text if opus produced nothing.
    return (opus or text), ("opus-mt-low-quality" if opus else "passthrough-failed")
