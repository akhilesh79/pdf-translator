# Quick Setup Checklist

✅ **Completed (Node.js side):**

- [x] Installed `python-shell` npm package
- [x] Created Python extraction script: `src/services/pdf_extractor.py`
- [x] Created Node.js Python wrapper: `src/services/pdfExtractorPython.js`
- [x] Updated `src/routes/translate.js` to use Python extractor

⏳ **Next Steps (Your installation):**

1. **Install Python 3.8+**

   ```bash
   # Windows: Download from https://www.python.org/downloads/
   # Make sure to check "Add Python to PATH"
   ```

2. **Install Tesseract OCR**

   ```bash
   # Windows: https://github.com/UB-Mannheim/tesseract/wiki
   # Default path: C:\Program Files\Tesseract-OCR
   ```

3. **Install Python packages**

   ```bash
   pip install pdfplumber pytesseract pillow pdf2image
   ```

4. **Verify setup**

   ```bash
   python --version
   python -c "import pdfplumber, pytesseract, pdf2image; print('OK')"
   ```

5. **Start your server**

   ```bash
   npm start
   ```

6. **Test with a PDF**
   - Upload a digital or scanned PDF
   - Should extract text accurately and translate it

## File Structure

```
src/
├── services/
│   ├── pdf_extractor.py          ← Python extraction script
│   ├── pdfExtractorPython.js      ← Node.js wrapper (NEW)
│   ├── languageDetector.js        ← Unchanged
│   └── translator.js              ← Unchanged
└── routes/
    └── translate.js               ← Updated to use Python extractor
```

## Key Benefits

- ✅ **Higher accuracy**: pdfplumber + pytesseract are industry-standard
- ✅ **Handles both**: Digital PDFs (pdfplumber) and scanned PDFs (OCR)
- ✅ **Better language support**: Tesseract recognizes 100+ languages
- ✅ **Automatic fallback**: Detects PDF type and uses appropriate method

**Issues?** Check:

- Python is in PATH: `python --version`
- Packages installed: `pip list | grep pdfplumber`
