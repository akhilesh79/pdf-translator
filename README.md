# PDF Translator

Upload a PDF in any language, get back the English translation. Handles both digital PDFs (direct text extraction via `pdfplumber`) and scanned/image-based PDFs (OCR via Tesseract). Translates using the free MyMemory API.

## Features

- **Any language in** — OCR via Tesseract covers 30+ scripts (Latin, Devanagari, Arabic, Han, Japanese, Korean, Cyrillic, Thai, Hebrew, etc.)
- **Auto script detection** — Tesseract OSD (Orientation and Script Detection) reads page 1 and picks the right OCR language pack automatically; no manual language hint needed
- **Digital + scanned PDFs** — `pdfplumber` for text-layer PDFs, falls back to OCR via `pdf2image` + `pytesseract` when the text layer is missing or unreliable
- **SHA-256 caching** — identical uploads are served from an in-memory LRU cache (24h TTL, 100 items)
- **Chunked translation with retries** — MyMemory's 500-char-per-request limit is handled by paragraph/sentence-aware chunking; network/5xx/429 errors retry with exponential backoff (1s→2s→4s→8s)
- **Quota-aware** — explicit `429 QUOTA_EXCEEDED` response when MyMemory's daily limit is reached
- **English passthrough** — English source text skips the translation API entirely

## API

### `POST /api/translate`

Multipart upload, field name `pdf`. Max file size 50 MB.

Optional `lang` field (form or query) overrides OCR language. Defaults to `auto` (OSD detection). Accepts Tesseract codes like `hin`, `ara`, `chi_sim`, or combos like `eng+hin`.

```bash
curl -F "pdf=@document.pdf" http://localhost:3000/api/translate
curl -F "pdf=@document.pdf" -F "lang=hin" http://localhost:3000/api/translate
```

**Success response:**
```json
{
  "success": true,
  "sourceLanguage": "hi",
  "isScanned": true,
  "ocrLang": "hin",
  "pageCount": 3,
  "engine": "mymemory",
  "originalText": "...",
  "translatedText": "...",
  "cached": false,
  "processingTimeMs": 4821
}
```

**Error codes:**
- `400` — no file, empty file, invalid PDF signature, invalid `lang` value
- `422` — PDF yielded no extractable text (image-only without supported lang pack, corrupt, or password-protected)
- `429` — `code: "QUOTA_EXCEEDED"` when MyMemory daily quota is exhausted
- `500` — extraction or translation pipeline error

### `GET /api/health`

Returns `{ success: true, ... }` for liveness checks.

## Local development

### Prerequisites

- **Node.js 20+**
- **Python 3.8+** (must be on PATH, or set `PYTHON_PATH`)
- **Tesseract OCR 5.x** with the `osd` language pack plus whichever scripts you need
- **Poppler** (required by `pdf2image` for scanned PDFs)

### Windows setup

1. Install Python from [python.org](https://www.python.org/downloads/) — tick "Add Python to PATH".
2. Install Tesseract from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Default path: `C:\Program Files\Tesseract-OCR`. During install, select the language packs you need **plus "osd"** (under Additional script data).
3. Download Poppler for Windows ([release archive](https://github.com/oschwartz10612/poppler-windows/releases)), unzip to e.g. `C:\poppler`, and set `POPPLER_PATH=C:\poppler\Library\bin` in `.env`.
4. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Install Node deps:
   ```bash
   npm install
   ```
6. Copy `.env.example` → `.env` and set `MYMEMORY_EMAIL` (optional but raises the free quota from 5K to 50K chars/day).
7. Run:
   ```bash
   npm run dev   # auto-reload
   npm start     # production mode
   ```

### Linux / macOS setup

```bash
# Debian/Ubuntu
sudo apt install python3 python3-pip poppler-utils \
    tesseract-ocr tesseract-ocr-osd tesseract-ocr-eng tesseract-ocr-hin

# macOS
brew install python poppler tesseract tesseract-lang

pip3 install -r requirements.txt
npm install
npm start
```

### Environment variables

| Var | Required | Purpose |
|---|---|---|
| `PORT` | no (default 3000) | HTTP port |
| `MYMEMORY_EMAIL` | no | Raises MyMemory free quota from ~5K to ~50K chars/day |
| `TESSERACT_PATH` | no | Path to `tesseract` binary — auto-detected from `PATH` otherwise |
| `POPPLER_PATH` | Windows only | Path to Poppler's `bin/` directory |
| `PYTHON_PATH` | no | Defaults to `python` (Win) or `python3` (Unix) |

## Deploying to Render

The repo includes a `Dockerfile`, `.dockerignore`, and `render.yaml` Blueprint.

1. Push to GitHub.
2. On [render.com](https://render.com): **New → Blueprint** → connect this repo. Render reads `render.yaml` and provisions a Docker web service.
3. In the service's **Environment** tab, set `MYMEMORY_EMAIL` (not committed because `sync: false`).
4. First build takes ~8–12 min (Tesseract language packs dominate). Subsequent deploys hit the layer cache and are much faster.

**Render free-tier caveats:**
- Container sleeps after 15 min idle → ~30s cold start.
- 512 MB RAM limit. Large scanned PDFs (50+ pages) may OOM during image conversion; drop DPI in `pdf_extractor.py` (300 → 200) or upgrade to Starter ($7/mo, 2 GB) if it becomes a problem.
- Image size ~1.3 GB with all 30 language packs. Trim the `tesseract-ocr-<lang>` lines in the `Dockerfile` for languages you don't need (each saves 5–20 MB).

## Project structure

```
src/
├── server.js                     Express app, startup Python-deps check
├── routes/
│   └── translate.js              Upload handler, validation, cache, orchestration
└── utils/
    ├── pdf_extractor.py          pdfplumber + OSD + Tesseract OCR
    ├── pdfExtractorPython.js     python-shell wrapper (captures stderr)
    ├── languageDetector.js       franc + English heuristic + ISO → Tesseract map
    ├── translator.js             MyMemory client (chunking, retries, quota handling)
    └── translationCache.js       LRU keyed by sha256(pdf) + ocrLang
Dockerfile                        Node 20 + Python 3 + poppler + 30 Tesseract langs
render.yaml                       Render Blueprint (Docker web service)
requirements.txt                  Python deps
```

## Troubleshooting

- **`Unable to get page count. Is poppler installed and in PATH?`** — Poppler missing. On Windows set `POPPLER_PATH`. On Linux install `poppler-utils`.
- **`UnicodeEncodeError: 'charmap' codec can't encode ...`** — Windows shell encoding issue; already handled (`PYTHONIOENCODING=utf-8` + `sys.stdout.reconfigure`). Reinstall deps if you still see it.
- **`Missing Tesseract language pack for <script>`** — OSD detected e.g. Devanagari but no `hin.traineddata` is installed. Install it: Windows → re-run Tesseract installer and tick the language; Linux → `apt install tesseract-ocr-hin`.
- **OCR returns gibberish Latin letters for non-Latin text** — the wrong language pack is being used. Check the `ocrLang` field in the response; if it says `eng` for Hindi text, OSD likely didn't detect the script (try passing `lang=hin` explicitly).
- **`QUOTA_EXCEEDED` (429)** — MyMemory daily limit hit. Set `MYMEMORY_EMAIL` to raise it, or wait until the quota resets.
