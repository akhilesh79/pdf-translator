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

## Test PDFs

The `test-pdfs/` folder contains sample PDF files used for testing the translation pipeline. These PDFs cover a variety of languages and formats, including:

- Digital PDFs (with selectable text)
- Scanned/image-based PDFs (requiring OCR)
- Multilingual documents (e.g., Hindi, Arabic, Chinese, Japanese, Russian, etc.)
- Mixed-script and multi-page PDFs

These test files help verify:

- Accurate text extraction (digital and scanned)
- Correct language/script detection
- Robustness of translation and error handling

You can add your own PDFs to this folder for custom testing. Example datasets may include:

- Public domain books in different languages
- Government forms or certificates
- Academic papers
- Synthetic PDFs generated for edge cases

**Note:** Do not include copyrighted or sensitive documents in this folder if sharing the repository.

## API

### `POST /api/translate`

Multipart upload, field name `pdf`. Max file size 50 MB.

Optional `lang` field (form or query) overrides OCR language. Defaults to `auto` (OSD detection). Accepts Tesseract codes like `hin`, `ara`, `chi_sim`, or combos like `eng+hin`.

```bash
curl -F "pdf=@document.pdf" https://pdf-translator-3p6o.onrender.com/api/translate
curl -F "pdf=@document.pdf" -F "lang=hin" https://pdf-translator-3p6o.onrender.com/api/translate
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
