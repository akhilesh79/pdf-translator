# Python Setup Guide

## Requirements

- **Python 3.10–3.12** — PyTorch wheels are only available for these versions. Python 3.13+ is not supported.
- **Poppler** — required by `pdf2image` to rasterise scanned PDF pages for OCR.

---

## Step 1: Install Python 3.10–3.12

### Windows

Download from https://www.python.org/downloads/ and install Python 3.12.

**Important:** Check "Add Python to PATH" during installation.

Verify:
```
python --version
```

### macOS

```bash
brew install python@3.12
```

### Linux (Ubuntu/Debian)

```bash
sudo apt install python3.12 python3.12-pip
```

---

## Step 2: Install Poppler

Poppler converts scanned PDF pages to images so Surya OCR can read them.

### Windows

```
winget install poppler
```

Or download from https://github.com/oschwartz10612/poppler-windows/releases, extract, and set `POPPLER_PATH` in `.env`:

```
POPPLER_PATH=C:\poppler\Library\bin
```

### macOS

```bash
brew install poppler
```

### Linux

```bash
sudo apt install poppler-utils
```

---

## Step 3: Install Python Packages

```bash
pip install -r requirements.txt
```

This installs (among others):

| Package | Purpose |
|---------|---------|
| `surya-ocr` | Transformer-based OCR — 90+ languages, handles handwriting and scanned documents |
| `torch` | Runs Surya OCR and opus-mt inference on CPU |
| `transformers` | Loads and runs the Helsinki-NLP/opus-mt-mul-en translation model |
| `pdfplumber` | Extracts text directly from digital PDFs |
| `pdf2image` | Converts scanned PDF pages to images via Poppler |
| `fastapi` + `uvicorn` | Python web service that Node.js talks to |
| `langdetect` | Detects source language from extracted text |

**Note:** `torch>=2.7.0,<2.11.0` is pinned — torch 2.11.0 has a known segfault with Surya's model loader on Windows CPU.

---

## Step 4: First Run — Model Downloads

On the first `npm start`, two model sets download automatically:

| Model | Cache Location | Size |
|-------|---------------|------|
| Surya OCR (detection + recognition) | `%LOCALAPPDATA%\datalab\datalab\Cache\models\` | ~1.5 GB |
| Helsinki-NLP/opus-mt-mul-en | `.hf_cache/` in project root | ~300 MB |

Total: ~1.8 GB. Allow 5–10 minutes on first run depending on connection speed. All subsequent starts load from disk in ~1 minute.

---

## Verify

```bash
python -c "import surya, fastapi, uvicorn, pdfplumber, langdetect, transformers, torch; print('All OK')"
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'torch'` | Run `pip install -r requirements.txt` |
| `PDF rasterization failed` | Poppler not installed or `POPPLER_PATH` not set in `.env` |
| `ModuleNotFoundError: No module named 'uvicorn'` | Wrong Python in PATH — set `PYTHON_PATH` in `.env` to the full interpreter path |
| `Service exited with code 3221225477` | torch version incompatibility — ensure `torch>=2.7.0,<2.11.0` is installed |
| Port 5000 already in use | Kill previous process: `netstat -ano \| findstr :5000` → `taskkill /PID <pid> /F` |
