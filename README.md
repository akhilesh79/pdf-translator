# PDF Translator

A REST API that extracts text from PDF and image files and translates it to English. Supports scanned documents, handwritten medical forms, printed reports, and insurance claim scans (PMJAY, BOCW, CMJAY) with mixed multilingual content — 90+ languages handled automatically.

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
                        ├── Surya OCR   — scanned PDF / image OCR (90+ languages, auto-detect)
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

**`lang` values:** Pass `auto` to let the system detect language automatically. Surya OCR is auto-multilingual so no language hint is needed for OCR. The `lang` field is used for the translation step.

**Response:**

```json
{
  "success": true,
  "sourceLanguage": "hi",
  "isScanned": true,
  "ocrLang": "auto",
  "pageCount": 1,
  "engine": "nllb-200",
  "originalText": "रोगी का नाम राजेश कुमार\nआयु 45 वर्ष...",
  "translatedText": "Patient Name: Rajesh Kumar\nAge 45 years...",
  "cached": false,
  "processingTimeMs": 120000
}
```

| Field | Description |
|-------|-------------|
| `isScanned` | `true` = Surya OCR was used; `false` = digital text extracted directly |
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
  "python": { "success": true, "surya_ready": true, "nllb_ready": true }
}
```

---

## Local Development

### Prerequisites

- Node.js 20+
- Python 3.11+ (3.14 supported with compatibility shim)
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

**First run:** Surya OCR models (~1.5 GB total) and NLLB-200 (~1.2 GB) download automatically into their respective caches. Allow 5–10 minutes on a good connection.

### Test with curl / Postman

```bash
# Health check
curl http://localhost:3000/health

# PDF (auto-detect language)
curl -X POST http://localhost:3000/api/translate \
  -F "file=@document.pdf"

# Image (language auto-detected by Surya)
curl -X POST http://localhost:3000/api/translate \
  -F "file=@scan.jpg"

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
| `HF_HOME` | `.hf_cache/` in project root | HuggingFace model cache (NLLB-200) |

---

## Docker

### Build

```bash
docker build -t pdf-translator .
```

Three-stage build:

| Stage | Base | What it does |
|-------|------|--------------|
| `py-deps` | `python:3.11-slim-bookworm` | Installs Python packages into `/opt/venv`; pre-bakes Surya weights |
| `node-deps` | `node:20-slim` | Installs Node production dependencies |
| `runtime` | `python:3.11-slim-bookworm` | Copies venv + node_modules + source; adds Node.js via NodeSource |

### Run

```bash
# Mount .hf_cache so NLLB-200 persists across container restarts
docker run -p 3000:3000 \
  -v "$(pwd)/.hf_cache:/app/.hf_cache" \
  pdf-translator
```

First start downloads NLLB-200 to the mounted volume (~2 min). All subsequent starts load from disk.

---

## Render Deployment

### 1. Create a Web Service

| Setting | Value |
|---------|-------|
| Runtime | **Docker** |
| Branch | `main` |
| Dockerfile path | `./Dockerfile` |

### 2. Plan

**Standard ($25/mo, 2 GB RAM)** minimum — Surya OCR + NLLB-200 require ~2.5 GB RAM at runtime. The free tier (512 MB) will OOM.

### 3. Persistent Disk

Prevents models from re-downloading on every redeploy:

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
| Surya OCR load (pre-baked in image) | ~20 s |
| NLLB-200 load from persistent disk | ~20 s |
| **Ready to serve** | **~70 s total** |

First-ever deploy only: Surya models + NLLB-200 download from HuggingFace Hub to the persistent disk (~5–10 min extra).

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
    ├── main.py            FastAPI app — loads Surya OCR + NLLB at startup
    ├── extractor.py       pdfplumber → Surya OCR fallback; image OCR
    ├── surya_compat.py    Compatibility shims for surya + transformers 4.57+
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
| Surya OCR (detection + recognition) | ~1.5 GB | OCR on scanned PDFs and images — 90+ languages, auto-multilingual |
| NLLB-200-distilled-600M | ~1.2 GB | Offline translation, 200 languages → English |

**Surya OCR advantages over EasyOCR:**
- Handles handwritten text, printed text, and mixed documents
- Auto-multilingual — no language hints required
- Recognises Indic scripts (Hindi, Gujarati, Marathi, Bengali, Tamil, Telugu, etc.) trained on real documents, not just synthetic fonts
- Better accuracy on low-quality scans, stamps, and mixed layouts

**Translation speed (CPU):** ~15–40 s per page (translation only). OCR adds ~2–6 min per dense scanned page on CPU. A GPU reduces both to under 30 s total.

**OCR accuracy:** 85–95% on typed/printed scans · 60–80% on handwritten text (significantly better than EasyOCR's 20–50%).
