import os
import sys
import multiprocessing

# Use a project-local cache dir (avoids Windows permission issues with ~/.cache)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# FLORES-200 codes used by NLLB-200.
# langdetect returns ISO 639-1 (e.g. "hi"); NLLB needs FLORES codes (e.g. "hin_Deva").
LANGDETECT_TO_NLLB = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "ar": "arb_Arab",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "uk": "ukr_Cyrl",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "ms": "zsm_Latn",
    "id": "ind_Latn",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    "km": "khm_Khmr",
}

TARGET_LANG = "eng_Latn"
# NLLB-200 has a 1024 token limit; ~300 words is safely under it.
MAX_CHUNK_WORDS = 300
# Max chunks to batch in a single model.generate() call (limits peak RAM).
MAX_BATCH_SIZE = 4


def _chunk_by_paragraph(text: str) -> list[str]:
    """Split on blank lines, then sub-split long paragraphs by sentence."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
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


def load_nllb_model() -> dict:
    """
    Download (first run) and load NLLB-200 distilled 600M with int8 quantization.
    Dynamic quantization converts Linear layers to int8 — 2-3x faster on CPU,
    no GPU needed, minimal quality loss.
    """
    import torch
    import torch.quantization
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Use all available CPU cores for matrix ops
    torch.set_num_threads(multiprocessing.cpu_count())

    model_name = "facebook/nllb-200-distilled-600M"
    print(f"[translator] Loading {model_name} (downloading on first run ~1.2 GB)...", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    # Dynamic int8 quantization: replaces nn.Linear weights with int8 at inference time.
    # No calibration data needed; applied once here so every request benefits.
    print("[translator] Applying int8 quantization for faster CPU inference...", file=sys.stderr)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    print(f"[translator] NLLB-200 ready (threads={torch.get_num_threads()}).", file=sys.stderr)
    return {"model": model, "tokenizer": tokenizer}


def _translate_batch(chunks: list[str], model, tokenizer, target_token_id: int) -> list[str]:
    """Translate a batch of chunks in a single model.generate() call."""
    import torch

    inputs = tokenizer(
        chunks,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_length=512,
            num_beams=1,
            do_sample=False,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]


def translate_to_english(text: str, source_lang: str, nllb_bundle: dict) -> tuple[str, str]:
    """
    Translate `text` to English using the locally-loaded NLLB-200 model.
    No external API — runs entirely offline after the first model download.
    Chunks are batched together so the model runs once per batch, not once per chunk.
    """
    if not text or not text.strip():
        return "", "noop"

    src_nllb = LANGDETECT_TO_NLLB.get(source_lang, "eng_Latn")

    if src_nllb == TARGET_LANG or source_lang == "en":
        return text, "passthrough"

    model = nllb_bundle["model"]
    tokenizer = nllb_bundle["tokenizer"]
    target_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    tokenizer.src_lang = src_nllb
    chunks = _chunk_by_paragraph(text)

    print(f"[translator] Translating {len(chunks)} chunk(s) from {src_nllb}...", file=sys.stderr)

    translated_parts: list[str] = []

    # Process in batches — fewer model.generate() calls = much faster
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i: i + MAX_BATCH_SIZE]
        try:
            results = _translate_batch(batch, model, tokenizer, target_token_id)
            translated_parts.extend(results)
        except Exception as e:
            print(f"[translator] batch {i} failed ({e}), falling back to per-chunk", file=sys.stderr)
            for chunk in batch:
                try:
                    results = _translate_batch([chunk], model, tokenizer, target_token_id)
                    translated_parts.append(results[0])
                except Exception as e2:
                    print(f"[translator] chunk failed ({e2}), keeping original", file=sys.stderr)
                    translated_parts.append(chunk)

    return "\n\n".join(translated_parts), "nllb-200"
