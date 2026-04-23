# PDF Translator — Complete Project Overview

This document explains every part of the project: what it does, why each library was chosen, how the code flows from an incoming request to the final response, and what each file is responsible for.

---

## What This Project Does

A user uploads a PDF or image file (scanned medical form, insurance claim, government document, handwritten notes). The API:
1. Extracts the text from the file (using OCR if the file is a scanned image or handwritten document)
2. Detects what language the text is in
3. Translates it to English
4. Returns the original text + translation in one JSON response

The entire pipeline runs **offline on the server** — no external APIs, no API keys, no internet required after the first model download.

---

## Why Two Servers? (Node.js + Python)

The project runs **two processes in one container**:

| Process | Language | Port | Role |
|---------|----------|------|------|
| Express | Node.js | 3000 | Accepts uploads, validates files, manages cache, returns API responses |
| FastAPI | Python | 5000 | Does the actual work: OCR, language detection, translation |

**Why not do everything in Node.js?**
OCR and AI translation libraries only exist in the Python ecosystem. Surya OCR, PyTorch, HuggingFace Transformers — none of these have Node.js equivalents.

**Why not do everything in Python?**
Python is great for ML but slow at web server boilerplate. Express handles file uploads, validation, routing, and caching cleanly in Node.js.

**How they connect:**
Node.js spawns the Python FastAPI process as a child process at startup. They talk over HTTP on localhost. Node acts as a proxy — receives the file from the client, forwards it to Python, gets back JSON, adds cache metadata, returns to client.

---

## Full Request Flow

```
User (Postman / frontend)
        │
        │  POST /api/translate  (multipart form: file + lang)
        ▼
┌─────────────────────────────────────────────────────┐
│                  Node.js  :3000                     │
│                                                     │
│  1. Multer saves file to uploads/ with unique name  │
│  2. Read file bytes → SHA-256 hash                  │
│  3. Check LRU cache  ──── HIT ──► return cached     │
│                      │                              │
│                    MISS                             │
│                      │                              │
│  4. POST multipart to Python :5000/process          │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────▼──────────────────────────────┐
        │              Python  :5000                  │
        │                                             │
        │  5. Is it a PDF or image?                   │
        │     PDF → pdfplumber (digital text)         │
        │         → Surya OCR fallback if scanned     │
        │     Image → Surya OCR directly              │
        │                                             │
        │  6. clean_ocr_text() — remove noise lines   │
        │  7. langdetect → detect source language     │
        │  8. NLLB-200 → translate to English         │
        │  9. Return JSON                             │
        └──────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Node.js  :3000                     │
│                                                     │
│  10. Store result in LRU cache                      │
│  11. Delete uploaded temp file                      │
│  12. Return JSON to user                            │
└─────────────────────────────────────────────────────┘
```

---

## File-by-File Breakdown

### `src/server.js` — Application Entry Point

This is the first file that runs (`node src/server.js`).

**What it does:**
- Creates the Express app
- Registers the `/api/translate` route
- Registers the `/health` route
- **Spawns the Python FastAPI service** as a child process
- **Polls Python's `/health` endpoint** every 2 seconds until it responds
- Only then starts listening on port 3000

**Why the polling loop?**
Surya OCR and NLLB-200 take 30–90 seconds to load on first run. If Node started accepting requests immediately, they would fail because Python isn't ready yet.

```
startPythonService()
    └── spawn: python3 -m uvicorn src.python.main:app --port 5000

waitForPythonReady()
    └── every 2s: GET http://127.0.0.1:5000/health
        └── when 200 OK: app.listen(3000)
```

---

### `src/routes/translate.js` — Upload Handler

Handles `POST /api/translate`. Validates, caches, and orchestrates — no ML work itself.

**1. Multer file filter** — checks MIME type and extension before saving.

**2. Filename sanitisation** — `{timestamp}-{8 random hex bytes}-{safe name}` prevents collisions.

**3. Lang validation** — validated against `/^[a-z0-9_+,]{2,60}$/` to block injection.

**4. Cache check** — SHA-256(file bytes) + lang = key. Hit → return immediately.

**5. Forward to Python** — multipart POST to FastAPI.

**6. Clean up** — temp file deleted in `finally` block regardless of success/failure.

---

### `src/utils/pythonClient.js` — HTTP Bridge to Python

- `forwardToPython()` — streams file + lang field to `http://127.0.0.1:5000/process`, 180 s timeout
- `checkPythonHealth()` — GETs `/health`, used for startup polling and the `/health` route

---

### `src/utils/translationCache.js` — LRU Cache

- `lru-cache` — 100 items max, 24 h TTL
- Key: `sha256(file_bytes):lang`
- Resets on Node process restart (in-memory only, not shared across instances)

---

### `src/python/main.py` — FastAPI Application

**Startup sequence:**
```python
async def lifespan(app):
    _patch_surya()                          # transformers compat shim
    foundation = FoundationPredictor()      # loads 1.4 GB recognition base
    models["surya"] = {
        "det_predictor": DetectionPredictor(),
        "rec_predictor": RecognitionPredictor(foundation_predictor=foundation),
    }
    models["nllb"] = load_nllb_model()
    yield
    models.clear()
```

**Routes:**
- `GET /health` — returns `{ surya_ready, nllb_ready }`
- `POST /process` — accepts `file` (UploadFile) + `lang` (Form)

---

### `src/python/surya_compat.py` — Compatibility Shims

Surya OCR 0.17.x relies on two `transformers` internals that changed in 4.57+:

| Issue | Root cause | Patch |
|-------|-----------|-------|
| `AttributeError: pad_token_id` | `PretrainedConfig` stopped auto-setting `pad_token_id=None` on instances | `SuryaDecoderConfig.pad_token_id = None` |
| `KeyError: 'default'` in `ROPE_INIT_FUNCTIONS` | The `'default'` rope type was removed from the dict | Inject a standard RoPE init function under `'default'` |

`apply()` is called once at startup before any Surya model is imported.

---

### `src/python/extractor.py` — The OCR + Extraction Engine

#### `is_pdf(file_bytes)`
Checks the first 5 bytes for `%PDF-`. More reliable than trusting the file extension.

#### `pdfplumber_looks_reliable(text)`
Decides if digital text extraction is trustworthy:
- At least 25 characters
- No `(cid:...)` garbage (embedded font without encoding)
- ≥ 55% of non-whitespace chars are alphanumeric

If any check fails → fall back to Surya OCR.

#### `clean_ocr_text(text)`

| Problem | Fix |
|---------|-----|
| Ligatures, half-width chars | `unicodedata.normalize("NFKC", text)` |
| Isolated single characters | Drop lines ≤ 2 chars that aren't alphabetic |
| Lines with <25% alphabetic ratio | Drop (scan artifacts like `~~~___~~~`) |
| Repeated punctuation | Regex collapse |
| Extra spaces around punctuation | Regex fix |
| Multiple consecutive blank lines | Collapse to single paragraph break |

**Why does cleaning matter?**
NLLB-200 is a translation model, not a text cleaner. Feeding it `~~~ 0 __ 0 ~~~\nPatient Name:` causes it to try to "translate" the noise. Cleaning first means the model only sees real text.

#### `_process_pdf` flow
```
pdfplumber extract text
    → reliable? → _build_result(isScanned=False)
    → not reliable? → pdf2image → PIL images → Surya OCR → _build_result(isScanned=True)
```

#### `ocr_images(surya_models, pil_images)`
Wraps the entire Surya batch call in a `ThreadPoolExecutor` with a 5-minute total timeout. All pages are passed to Surya in a single batch call (more efficient than one-by-one).

#### `_run_surya(surya_models, pil_images)`
```python
results = rec_predictor(pil_images, det_predictor=det_predictor)
# results[i].text_lines[j].text → recognised string for each text region
```

---

### `src/python/detector.py` — Language Detection

Uses `langdetect` (Naive Bayes on character n-grams) on the first 4000 chars of cleaned text. Returns ISO 639-1 codes (`"hi"`, `"en"`, `"fr"`, etc.) which are then mapped to FLORES-200 codes for NLLB.

---

### `src/python/translator.py` — NLLB-200 Translation Engine

#### Why NLLB-200?

| Option | Problem |
|--------|---------|
| Google Translate API | Costs money, needs internet, API key |
| MyMemory API | 5K chars/day limit, needs internet |
| Helsinki-NLP opus-mt | One model per language pair — 50+ models needed |
| **NLLB-200** | One model, 200 languages, fully offline, free forever |

#### Key implementation details

**Dynamic int8 quantization** — converts `nn.Linear` weights from fp32 → int8, reducing memory from ~2.4 GB to ~800 MB and speeding up inference 2–3× on CPU.

**`_chunk_by_paragraph(text)`** — splits text into ≤ 300-word chunks to fit NLLB's 1024-token limit.

**Batched inference** — all chunks passed to `model.generate()` in one call. Sequential: 3 × 35s = 105s. Batched: ~40–50s.

**`num_beams=1`** — greedy decoding. Near-identical quality to beam search at 4× the speed for factual text.

**`forced_bos_token_id`** — forces the output to start with the English language token `eng_Latn`, locking output to English regardless of input language.

---

### `src/python/models.py` — Response Schema

Pydantic model used by FastAPI to validate and serialise the `process_file()` response. All fields are `Optional` because error responses only return `success` + `error`.

---

## Library Glossary

### Node.js

| Library | Why it's used |
|---------|--------------|
| **express** | Web framework — routing, middleware, JSON responses |
| **multer** | `multipart/form-data` file uploads, streams to disk |
| **axios** | HTTP client for Node → Python forwarding |
| **form-data** | Constructs multipart form for Node → Python |
| **lru-cache** | In-memory LRU cache — avoids re-processing identical files |
| **dotenv** | Loads `.env` into `process.env` |

### Python

| Library | Why it's used |
|---------|--------------|
| **fastapi** | Async web framework with auto JSON serialisation + OpenAPI docs |
| **uvicorn** | ASGI server that runs FastAPI |
| **python-multipart** | FastAPI dependency for parsing file uploads |
| **pdfplumber** | Extracts text from digital PDFs |
| **pdf2image** | Converts scanned PDF pages to PIL images via Poppler |
| **Pillow (PIL)** | Image loading and RGB conversion |
| **surya-ocr** | Transformer-based OCR — 90+ languages, auto-multilingual, handles handwriting, printed text, mixed scripts |
| **langdetect** | Fast offline language detection (55 languages, Naive Bayes) |
| **transformers** | HuggingFace library — provides NLLB-200 model + tokenizer |
| **torch (PyTorch)** | Runs Surya and NLLB-200 inference, int8 quantization |
| **sentencepiece** | Tokenizer required by NLLB-200 |
| **sacremoses** | Text normalisation required by NLLB-200 tokenizer |
| **pydantic-settings** | Required by surya-ocr for configuration |
| **platformdirs** | Required by surya-ocr for cross-platform model cache paths |

---

## Model Storage

| Model | Cache Location | Size |
|-------|---------------|------|
| Surya detection | `%LOCALAPPDATA%\datalab\datalab\Cache\models\text_detection\` | ~74 MB |
| Surya recognition | `%LOCALAPPDATA%\datalab\datalab\Cache\models\text_recognition\` | ~1.4 GB |
| NLLB-200-distilled-600M | `.hf_cache/hub/models--facebook--nllb-200-distilled-600M/` | ~1.2 GB |

---

## Environment Variables Reference

| Variable | Default | Effect |
|----------|---------|--------|
| `PORT` | `3000` | Node.js listen port |
| `PYTHON_PATH` | `python3` / `python` | Python interpreter Node spawns |
| `PYTHON_SERVICE_URL` | `http://127.0.0.1:5000` | FastAPI base URL |
| `HF_HOME` | `.hf_cache/` in project root | NLLB-200 model cache directory |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | Suppresses Windows symlink warning |
| `POPPLER_PATH` | None | Path to Poppler binaries on Windows |

---

## Startup Sequence

```
npm run dev
    │
    ▼
node src/server.js
    │
    ├── spawn("python3 -m uvicorn src.python.main:app --port 5000")
    │       │
    │       ▼
    │   lifespan() runs:
    │       ├── surya_compat.apply()       ← transformers compat patches
    │       ├── DetectionPredictor()       ← 74 MB text detection model
    │       ├── FoundationPredictor()      ← 1.4 GB recognition base model
    │       ├── RecognitionPredictor(...)  ← wires detection + recognition
    │       └── load_nllb_model()
    │               ├── AutoTokenizer + AutoModelForSeq2SeqLM
    │               └── quantize_dynamic() ← int8 optimisation
    │
    ├── every 2s: GET /health  ... waiting ~20–60s ...
    │
    └── app.listen(3000) → "PDF Translator API running on port 3000"
```

Total startup: **20–60 s** (models cached from second run onward).

---

## Performance

| Scenario | Typical Time |
|----------|-------------|
| Digital PDF, any language, 1 page | 15–40 s (translation only) |
| Scanned PDF — printed text, 1 page | 2–4 min (Surya + translation) |
| Scanned PDF — handwritten, 1 page | 3–6 min (Surya + translation) |
| English document | < 1 s (passthrough, no translation) |
| Same file uploaded again | < 50 ms (LRU cache hit) |

**Bottleneck:** Surya recognition runs each detected text region through the transformer sequentially on CPU. GPU reduces this to seconds.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `timeout of 180000ms exceeded` | Dense page took > 3 min on CPU | Increase timeout in `pythonClient.js` or reduce image size |
| `PDF rasterization failed` | Poppler not installed or not on PATH | Install Poppler, set `POPPLER_PATH` in `.env` |
| `No text could be extracted` | Blank scan or all lines filtered as noise | Check scan quality |
| `AttributeError: pad_token_id` | surya-ocr + transformers compat issue | Fixed by `surya_compat.apply()` — ensure it runs before any Surya import |
| `KeyError: 'default'` | surya-ocr + transformers compat issue | Fixed by `surya_compat.apply()` |
| `only one usage of each socket address` | Previous uvicorn process holding port 5000 | `netstat -ano \| findstr :5000` → kill the PID |
| `HF_HOME permission denied` | HuggingFace wrote to restricted path | Set `HF_HOME` to project folder |
