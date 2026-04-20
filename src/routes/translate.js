const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const fsp = require('fs').promises;
const crypto = require('crypto');

const { extractTextPython } = require('../utils/pdfExtractorPython');
const { detectLanguage } = require('../utils/languageDetector');
const { translateToEnglish, QuotaExceededError } = require('../utils/translator');
const cache = require('../utils/translationCache');

const router = express.Router();

const UPLOAD_DIR = path.resolve('./uploads');
const MAX_FILE_SIZE_MB = 50;
const DEFAULT_OCR_LANG = 'auto';
const LANG_RE = /^[a-z0-9_+]{2,40}$/;

if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

async function safeUnlink(filePath) {
  try {
    await fsp.unlink(filePath);
  } catch (err) {
    if (err.code !== 'ENOENT') {
      console.error(`[translate] Failed to delete ${filePath}: ${err.message}`);
    }
  }
}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const unique = `${Date.now()}-${crypto.randomBytes(8).toString('hex')}`;
    const safeOriginal = path.basename(file.originalname).replace(/[^a-zA-Z0-9._-]/g, '_');
    cb(null, `${unique}-${safeOriginal}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const okMime = file.mimetype === 'application/pdf';
    const okExt = path.extname(file.originalname).toLowerCase() === '.pdf';
    if (!okMime || !okExt) {
      return cb(new Error('Only PDF files are accepted'));
    }
    cb(null, true);
  },
});

async function isValidPdf(buffer) {
  if (!buffer || buffer.length < 5) return false;
  return buffer.slice(0, 5).toString('ascii') === '%PDF-';
}

router.post('/', upload.single('pdf'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      error: 'No PDF file uploaded. Use form field name "pdf".',
    });
  }

  const filePath = req.file.path;
  const startTime = Date.now();

  try {
    // 0. Read + sanity-check the file.
    const buffer = await fsp.readFile(filePath);
    if (buffer.length === 0) {
      return res.status(400).json({ success: false, error: 'Uploaded file is empty.' });
    }
    if (!(await isValidPdf(buffer))) {
      return res.status(400).json({
        success: false,
        error: 'Uploaded file is not a valid PDF (missing %PDF- signature).',
      });
    }

    // 1. Resolve OCR language: client override (form/query) > default (auto).
    let ocrLangArg = (req.body && req.body.lang) || req.query.lang || DEFAULT_OCR_LANG;
    ocrLangArg = String(ocrLangArg).toLowerCase().trim();
    if (!LANG_RE.test(ocrLangArg)) {
      return res.status(400).json({
        success: false,
        error: `Invalid lang value. Use 'auto', a Tesseract code like 'hin', or a combo like 'eng+hin'.`,
      });
    }

    // 2. Cache lookup by SHA-256 + lang (same PDF + different lang = different result).
    const cacheKey = `${cache.hashBuffer(buffer)}:${ocrLangArg}`;
    const cached = cache.get(cacheKey);
    if (cached) {
      return res.json({
        ...cached,
        cached: true,
        processingTimeMs: Date.now() - startTime, 
      });
    }

    // 3. Extract. Python handles script detection via OSD when lang='auto'.
    const { text, isScanned, pageCount, ocrLang } = await extractTextPython(filePath, {
      lang: ocrLangArg,
    });

    if (!text || text.trim().length === 0) {
      return res.status(422).json({
        success: false,
        error:
          'Could not extract any text. The PDF may be empty, image-only without OCR support, corrupt, or password-protected.',
      });
    }

    // 4. Detect source language (with English heuristic fallback).
    const sourceLanguage = detectLanguage(text);

    // 5. Translate (short-circuits to passthrough when source == en).
    const { translatedText, engine } = await translateToEnglish(text, sourceLanguage);

    const payload = {
      success: true,
      sourceLanguage,
      pageCount: pageCount || null,
      engine,
      originalText: text,
      translatedText,
    };

    cache.set(cacheKey, payload);

    return res.json({
      ...payload,
      cached: false,
      processingTimeMs: Date.now() - startTime,
    });
  } catch (err) {
    console.error('[translate] Failed:', err);
    if (res.headersSent) return undefined;

    if (err instanceof QuotaExceededError) {
      return res.status(429).json({
        success: false,
        code: 'QUOTA_EXCEEDED',
        error: err.message,
      });
    }
    return res.status(500).json({
      success: false,
      error: err.message || 'Internal error during PDF processing.',
    });
  } finally {
    safeUnlink(filePath);
  }
});

module.exports = router;
