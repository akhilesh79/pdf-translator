const { PythonShell } = require('python-shell');
const path = require('path');

const PYTHON_SCRIPT = path.resolve(__dirname, 'pdf_extractor.py');
const PYTHON_PATH = process.env.PYTHON_PATH
  || (process.platform === 'win32' ? 'python' : 'python3');
const EXTRACTION_TIMEOUT_MS = 180000; // 3 minutes

function runPython(args) {
  return new Promise((resolve, reject) => {
    const shell = new PythonShell(PYTHON_SCRIPT, {
      pythonPath: PYTHON_PATH,
      args,
      mode: 'text',
      encoding: 'utf8',
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });

    const stdoutLines = [];
    const stderrLines = [];

    shell.on('message', (msg) => stdoutLines.push(msg));
    shell.on('stderr', (msg) => stderrLines.push(msg));

    const timer = setTimeout(() => {
      shell.kill();
      reject(new Error(`PDF extraction timeout after ${EXTRACTION_TIMEOUT_MS}ms`));
    }, EXTRACTION_TIMEOUT_MS);

    shell.end((err) => {
      clearTimeout(timer);

      if (stderrLines.length) {
        console.error('[pdfExtractorPython] stderr:\n' + stderrLines.join('\n'));
      }

      if (err) {
        const detail = stderrLines.length ? `\n${stderrLines.join('\n')}` : '';
        return reject(new Error(`Python process failed: ${err.message}${detail}`));
      }

      const output = stdoutLines.join('\n').trim();
      if (!output) {
        return reject(new Error('Python script produced no output'));
      }

      try {
        resolve(JSON.parse(output));
      } catch (parseErr) {
        reject(new Error(
          `Failed to parse Python output: ${parseErr.message}\nRaw: ${output.slice(0, 500)}`,
        ));
      }
    });
  });
}

async function extractTextPython(filePath, { lang = 'auto' } = {}) {
  const data = await runPython([filePath, '--lang', lang]);

  if (!data.success) {
    throw new Error(data.error || 'PDF extraction failed (no error provided)');
  }

  return {
    text: data.text || '',
    isScanned: Boolean(data.isScanned),
    pageCount: data.pageCount || null,
    ocrLang: data.ocrLang || null,
  };
}

async function verifyPythonDeps() {
  try {
    const results = await PythonShell.runString(
      'import pdfplumber, pytesseract, pdf2image; print("ok")',
      { pythonPath: PYTHON_PATH },
    );
    if (!results || results[0] !== 'ok') {
      throw new Error('Dependency check returned unexpected output');
    }
  } catch (err) {
    throw new Error(
      `Python dependency check failed. Ensure '${PYTHON_PATH}' resolves and `
      + `pdfplumber, pytesseract, pdf2image are installed.\n${err.message}`,
    );
  }
}

module.exports = { extractTextPython, verifyPythonDeps };
