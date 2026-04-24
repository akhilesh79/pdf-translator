import os
import sys
import multiprocessing

# Use a project-local cache dir (avoids Windows permission issues with ~/.cache)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
# ~450 tokens is safely under the MarianMT 512-token limit at ~300 words.
MAX_CHUNK_WORDS = 300
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
    """Load Helsinki-NLP/opus-mt-mul-en (MarianMT, ~78M params, CPU-friendly)."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    torch.set_num_threads(multiprocessing.cpu_count())
    print(f"[translator] Loading {MODEL_NAME} on CPU...", file=sys.stderr)

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("[translator] Translation model ready (CPU).", file=sys.stderr)
    return {"model": model, "tokenizer": tokenizer, "device": "cpu"}


def _translate_batch(chunks: list[str], model, tokenizer) -> list[str]:
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
            max_length=512,
            num_beams=1,
            do_sample=False,
        )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]


def translate_to_english(text: str, source_lang: str, nllb_bundle: dict) -> tuple[str, str]:
    if not text or not text.strip():
        return "", "noop"

    if source_lang == "en":
        return text, "passthrough"

    model = nllb_bundle["model"]
    tokenizer = nllb_bundle["tokenizer"]

    chunks = _chunk_by_paragraph(text)
    print(f"[translator] Translating {len(chunks)} chunk(s) from {source_lang}...", file=sys.stderr)

    translated_parts: list[str] = []

    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i: i + MAX_BATCH_SIZE]
        try:
            results = _translate_batch(batch, model, tokenizer)
            translated_parts.extend(results)
        except Exception as e:
            print(f"[translator] batch {i} failed ({e}), falling back to per-chunk", file=sys.stderr)
            for chunk in batch:
                try:
                    results = _translate_batch([chunk], model, tokenizer)
                    translated_parts.append(results[0])
                except Exception as e2:
                    print(f"[translator] chunk failed ({e2}), keeping original", file=sys.stderr)
                    translated_parts.append(chunk)

    return "\n\n".join(translated_parts), "opus-mt"
