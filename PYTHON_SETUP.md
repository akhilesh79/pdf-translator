# PDF Translator - Python Setup Guide

This application now uses Python for accurate PDF text extraction with OCR support. Follow these steps to set up Python.

## Step 1: Install Python

### On Windows (Recommended)

1. Download Python from https://www.python.org/downloads/
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Verify installation by opening Command Prompt and running:
   ```
   python --version
   ```

### On macOS

```bash
brew install python3
```

### On Linux (Ubuntu/Debian)

```bash
sudo apt-get install python3 python3-pip
```

## Step 2: Install Tesseract OCR

### On Windows

1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (use default installation path: `C:\Program Files\Tesseract-OCR`)
3. The script will automatically detect it

### On macOS

```bash
brew install tesseract
```

### On Linux

```bash
sudo apt-get install tesseract-ocr
```

## Step 3: Install Python Packages

In the project directory, run:

```bash
pip install pdfplumber pytesseract pillow pdf2image
```

Or use:

```bash
pip3 install pdfplumber pytesseract pillow pdf2image
```

## Step 4: Verify Setup

After installing, test with:

```bash
python3 -c "import pdfplumber; import pytesseract; import pdf2image; print('All packages installed successfully!')"
```

## Troubleshooting

### Tesseract not found on Windows

If you get "tesseract is not installed" error, update the path in `src/services/pdf_extractor.py`:

```python
pytesseract.pytesseract.pytesseract_cmd = r'C:\Path\To\Your\Tesseract\tesseract.exe'
```

### pdfplumber errors

Make sure you have Poppler installed:

- **Windows**: Comes with pdf2image
- **macOS**: `brew install poppler`
- **Linux**: `sudo apt-get install poppler-utils`

### Python not found in Node.js

Make sure Python is in your system PATH and restart your terminal/IDE after installation.

## How It Works

1. User uploads PDF
2. `translate.js` calls `extractTextPython()`
3. Python script (`pdf_extractor.py`):
   - Tries `pdfplumber` first (for digital PDFs)
   - Falls back to OCR with `pytesseract` (for scanned PDFs)
   - Returns JSON with extracted text, language, and page count
4. Node.js processes the text for translation

## Performance Notes

- **Digital PDFs**: ~1-2 seconds per page (pdfplumber)
- **Scanned PDFs**: ~5-10 seconds per page (OCR - slower)
- **Mixed PDFs**: Automatic detection and appropriate method used

## Next Steps

After Python setup, start your server:

```bash
npm start
```

Upload a PDF via the API to test!
