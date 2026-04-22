const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:5000';
const REQUEST_TIMEOUT_MS = 180_000;

const MIME_MAP = {
  '.pdf':  'application/pdf',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png':  'image/png',
  '.tiff': 'image/tiff',
  '.tif':  'image/tiff',
};

function detectMime(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return MIME_MAP[ext] || 'application/octet-stream';
}

async function forwardToPython(filePath, { lang = 'auto' } = {}) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath), {
    filename: path.basename(filePath),
    contentType: detectMime(filePath),
  });
  form.append('lang', lang);

  const response = await axios.post(`${PYTHON_SERVICE_URL}/process`, form, {
    headers: form.getHeaders(),
    timeout: REQUEST_TIMEOUT_MS,
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
  });

  return response.data;
}

async function checkPythonHealth() {
  const response = await axios.get(`${PYTHON_SERVICE_URL}/health`, { timeout: 5000 });
  return response.data;
}

module.exports = { forwardToPython, checkPythonHealth };
