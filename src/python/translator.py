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
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(multiprocessing.cpu_count())

    model_name = "facebook/nllb-200-distilled-600M"
    print(f"[translator] Loading {model_name} on {device.upper()}...", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device == "cuda":
        # fp16 on GPU: ~600 MB VRAM, 10-20x faster than CPU int8
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model = model.to(device)
    else:
        # int8 on CPU: 2-3x faster than fp32, no GPU needed
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    model.eval()
    print(f"[translator] NLLB-200 ready ({device.upper()}).", file=sys.stderr)
    return {"model": model, "tokenizer": tokenizer, "device": device}


def _translate_batch(chunks: list[str], model, tokenizer, target_token_id: int, device: str = "cpu") -> list[str]:
    import torch

    inputs = tokenizer(
        chunks,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

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
    if not text or not text.strip():
        return "", "noop"

    src_nllb = LANGDETECT_TO_NLLB.get(source_lang, "eng_Latn")

    if src_nllb == TARGET_LANG or source_lang == "en":
        return text, "passthrough"

    model = nllb_bundle["model"]
    tokenizer = nllb_bundle["tokenizer"]
    device = nllb_bundle.get("device", "cpu")
    target_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    tokenizer.src_lang = src_nllb
    chunks = _chunk_by_paragraph(text)

    print(f"[translator] Translating {len(chunks)} chunk(s) from {src_nllb}...", file=sys.stderr)

    translated_parts: list[str] = []

    # Process in batches — fewer model.generate() calls = much faster
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i: i + MAX_BATCH_SIZE]
        try:
            results = _translate_batch(batch, model, tokenizer, target_token_id, device)
            translated_parts.extend(results)
        except Exception as e:
            print(f"[translator] batch {i} failed ({e}), falling back to per-chunk", file=sys.stderr)
            for chunk in batch:
                try:
                    results = _translate_batch([chunk], model, tokenizer, target_token_id, device)
                    translated_parts.append(results[0])
                except Exception as e2:
                    print(f"[translator] chunk failed ({e2}), keeping original", file=sys.stderr)
                    translated_parts.append(chunk)

    return "\n\n".join(translated_parts), "nllb-200"
