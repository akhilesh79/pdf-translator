# PDF Translator

A REST API that extracts text from PDF and image files and translates it to English. Supports scanned documents, printed medical forms, and insurance claim scans (PMJAY, BOCW) with mixed Hindi/English/Gujarati content.

**No external API keys required** — OCR and translation run entirely offline using locally-loaded models.

---

## Architecture

```
Client
  │
  ▼
Node.js :3000  (Express)
  ├── Multer       — file upload (PDF, JPEG, PNG, TIFF, max 50 MB)
  ├── LRU cache    — SHA-256 key, 100 items, 24 h TTL
  └── HTTP POST ──► Python :5000  (FastAPI)
                        ├── pdfplumber  — digital PDF text extraction
                        ├── EasyOCR     — scanned PDF / image OCR (hi + en)
                        └── NLLB-200    — offline translation, 200 languages → English
```

Node.js spawns the FastAPI service at startup and waits for `/health` before accepting requests.

---

## API

### `POST /api/translate`

Upload a PDF or image. Returns extracted text and its English translation.

**Body:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | PDF, JPEG, PNG, or TIFF |
| `lang` | String | No | Source language hint (default: `auto`) |

**`lang` values:** `hi` Hindi · `gu` Gujarati · `mr` Marathi · `pa` Punjabi · `bn` Bengali · `ta` Tamil · `te` Telugu · `kn` Kannada · `ml` Malayalam · `ar` Arabic · `fr` French · `de` German · `es` Spanish · `en` English (passthrough) · `auto` detect automatically

**Response:**

```json
{
  "success": true,
  "sourceLanguage": "hi",
  "isScanned": false,
  "ocrLang": null,
  "pageCount": 1,
  "engine": "nllb-200",
  "originalText": "भारत दक्षिण एशिया में स्थित एक विशाल देश है...",
  "translatedText": "India is a vast country located in South Asia...",
  "cached": false,
  "processingTimeMs": 35000
}
```

| Field | Description |
|-------|-------------|
| `isScanned` | `true` = EasyOCR was used; `false` = digital text extracted directly |
| `engine` | `nllb-200` translated · `passthrough` already English · `noop` empty |
| `cached` | `true` = served from LRU cache (same file + lang uploaded before) |

**Error codes:**

| Code | Reason |
|------|--------|
| `400` | No file, empty file, disallowed type, or invalid `lang` |
| `422` | No text could be extracted (corrupt, password-protected, or blank scan) |
| `500` | Unexpected pipeline error |

### `GET /health`

```json
{
  "success": true,
  "message": "Server Health is Awesome!",
  "python": { "success": true, "easyocr_ready": true, "nllb_ready": true }
}
```

---

## Local Development

### Prerequisites

- Node.js 20+
- Python 3.11+
- Poppler (PDF → image for scanned PDFs)

```bash
# Windows
winget install poppler

# macOS
brew install poppler

# Ubuntu / Debian
sudo apt install poppler-utils
```

### Setup

```bash
# Python deps
pip install -r requirements.txt

# Node deps
npm install

# Start — Node auto-spawns the FastAPI service
npm run dev
```

**First run:** EasyOCR (~200 MB) and NLLB-200 (~1.2 GB) download automatically into `.hf_cache/`. Allow 3–5 minutes on a good connection.

### Test with curl / Postman

```bash
# Health check
curl http://localhost:3000/health

# PDF (auto-detect language)
curl -X POST http://localhost:3000/api/translate \
  -F "file=@document.pdf"

# Image with language hint
curl -X POST http://localhost:3000/api/translate \
  -F "file=@scan.jpg" \
  -F "lang=hi"

# Hit FastAPI directly (bypasses Node LRU cache)
curl -X POST http://localhost:5000/process \
  -F "file=@document.pdf" \
  -F "lang=auto"
```

**Postman:** Body → form-data → key `file` (type: File) → select file. Optionally add key `lang` (type: Text).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Node.js listen port |
| `PYTHON_PATH` | `python3` / `python` | Python interpreter |
| `PYTHON_SERVICE_URL` | `http://127.0.0.1:5000` | FastAPI base URL |
| `HF_HOME` | `.hf_cache/` in project root | HuggingFace model cache |

---

## Docker

### Build

```bash
docker build -t pdf-translator .
```

Three-stage build:

| Stage | Base | What it does |
|-------|------|--------------|
| `py-deps` | `python:3.11-slim-bookworm` | Installs Python packages into `/opt/venv`; pre-bakes EasyOCR weights |
| `node-deps` | `node:20-slim` | Installs Node production dependencies |
| `runtime` | `python:3.11-slim-bookworm` | Copies venv + node_modules + source; adds Node.js via NodeSource |

Each stage is an independent cache layer — changing `package.json` only invalidates the Node stage.

### Run

```bash
# Mount .hf_cache so NLLB-200 persists across container restarts
docker run -p 3000:3000 \
  -v "$(pwd)/.hf_cache:/app/.hf_cache" \
  pdf-translator
```

First start downloads NLLB-200 to the mounted volume (~2 min). All subsequent starts load it instantly from disk.

---

## Render Deployment

### 1. Create a Web Service

| Setting | Value |
|---------|-------|
| Runtime | **Docker** |
| Branch | `main` |
| Dockerfile path | `./Dockerfile` |

### 2. Plan

**Standard ($25/mo, 2 GB RAM)** minimum — NLLB-200 requires ~1.5 GB at runtime. The free tier (512 MB) will OOM.

### 3. Persistent Disk

Prevents NLLB-200 from re-downloading on every redeploy:

| Setting | Value |
|---------|-------|
| Name | `hf-cache` |
| Mount Path | `/app/.hf_cache` |
| Size | 5 GB |

### 4. Environment Variables

| Key | Value |
|-----|-------|
| `PYTHON_PATH` | `/opt/venv/bin/python3` |

`PORT` is injected automatically by Render.

### 5. Health Check

In **Settings → Health & Alerts** set the path to `/health`.

### Cold Start Timeline

| Event | Duration |
|-------|----------|
| Image pull + container start | ~30 s |
| EasyOCR load (pre-baked in image) | ~10 s |
| NLLB-200 load from persistent disk | ~20 s |
| **Ready to serve** | **~60 s total** |

First-ever deploy only: NLLB-200 downloads from HuggingFace Hub to the persistent disk (~2–3 min extra).

---

## Project Structure

```
src/
├── server.js              Express app — spawns FastAPI, waits for /health
├── routes/
│   └── translate.js       Upload handler, validation, LRU cache, pythonClient call
├── utils/
│   ├── pythonClient.js    HTTP client: POST multipart to FastAPI :5000
│   └── translationCache.js  LRU cache keyed by sha256(file) + lang
└── python/
    ├── main.py            FastAPI app — loads EasyOCR + NLLB at startup
    ├── extractor.py       pdfplumber → EasyOCR fallback; image OCR
    ├── translator.py      NLLB-200 with int8 quantization + batched inference
    ├── detector.py        langdetect language detection
    └── models.py          Pydantic response schema

Dockerfile                 3-stage build (py-deps / node-deps / runtime)
requirements.txt           Python deps
package.json               Node deps
```

---

## Model Details

| Model | Size | Used for |
|-------|------|----------|
| EasyOCR (hi, en) | ~200 MB | OCR on scanned PDFs and images |
| NLLB-200-distilled-600M | ~1.2 GB | Offline translation, 200 languages → English |

**Translation speed (CPU):** ~15–40 s per page. int8 dynamic quantization and batched inference reduce latency 2–3× versus the baseline.

**OCR accuracy:** 70–90% on typed/printed scans · 20–50% on handwritten text.
