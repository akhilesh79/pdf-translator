const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const fsp = require('fs').promises;
const crypto = require('crypto');

const { forwardToPython } = require('../utils/pythonClient');
const cache = require('../utils/translationCache');

const router = express.Router();

const UPLOAD_DIR = path.resolve('./uploads');
const MAX_FILE_SIZE_MB = 50;
const DEFAULT_LANG = 'auto';
const LANG_RE = /^[a-z0-9_+,]{2,60}$/;

const ALLOWED_MIMES = new Set([
  'application/pdf',
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/tiff',
]);
const ALLOWED_EXTS = new Set(['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']);

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
    const okMime = ALLOWED_MIMES.has(file.mimetype);
    const okExt = ALLOWED_EXTS.has(path.extname(file.originalname).toLowerCase());
    if (!okMime || !okExt) {
      return cb(new Error('Only PDF and image files (JPEG, PNG, TIFF) are accepted.'));
    }
    cb(null, true);
  },
});

router.post('/', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      error: 'No file uploaded. Use form field name "file" (PDF, JPEG, PNG, or TIFF).',
    });
  }

  const filePath = req.file.path;
  const startTime = Date.now();

  try {
    const buffer = await fsp.readFile(filePath);
    if (buffer.length === 0) {
      return res.status(400).json({ success: false, error: 'Uploaded file is empty.' });
    }

    let langArg = (req.body && req.body.lang) || req.query.lang || DEFAULT_LANG;
    langArg = String(langArg).toLowerCase().trim();
    if (!LANG_RE.test(langArg)) {
      return res.status(400).json({
        success: false,
        error: `Invalid lang value. Use 'auto', a language code like 'hi', 'en', 'ar', or combos like 'hi+en'.`,
      });
    }

    const cacheKey = `${cache.hashBuffer(buffer)}:${langArg}`;
    const cached = cache.get(cacheKey);
    if (cached) {
      return res.json({ ...cached, cached: true, processingTimeMs: Date.now() - startTime });
    }

    const payload = await forwardToPython(filePath, { lang: langArg });

    cache.set(cacheKey, payload);

    return res.json({ ...payload, cached: false, processingTimeMs: Date.now() - startTime });
  } catch (err) {
    console.error('[translate] Failed:', err);
    if (res.headersSent) return undefined;

    const status = err.response?.status;
    const data = err.response?.data;

    if (status === 422) {
      return res.status(422).json({ success: false, error: data?.detail || 'Could not extract text from file.' });
    }
    return res.status(500).json({
      success: false,
      error: err.message || 'Internal error during file processing.',
    });
  } finally {
    safeUnlink(filePath);
  }
});

module.exports = router;
