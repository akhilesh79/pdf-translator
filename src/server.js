require('dotenv').config();
const { spawn, execSync } = require('child_process');
const express = require('express');
const axios = require('axios');
const translateRouter = require('./routes/translate');
const { checkPythonHealth } = require('./utils/pythonClient');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_PATH = process.env.PYTHON_PATH || (process.platform === 'win32' ? 'python' : 'python3');
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:5000';

app.use(express.json({ limit: '50mb' }));
app.use('/api/translate', translateRouter);

app.get('/health', async (req, res) => {
  try {
    const pyHealth = await checkPythonHealth();
    res.json({ success: true, message: 'Server Health is Awesome!', python: pyHealth });
  } catch {
    res.status(503).json({ success: false, message: 'Python service unavailable' });
  }
});

app.use((req, res) => {
  res.status(404).json({ success: false, error: 'Route not found' });
});

app.use((err, req, res, next) => {
  console.error('Handling error:', err.message);
  return res.status(500).json({ success: false, error: err.message });
});

function killPort5000() {
  try {
    if (process.platform === 'win32') {
      const out = execSync('netstat -ano', { encoding: 'utf8', timeout: 5000 });
      const match = out.match(/127\.0\.0\.1:5000\s+\S+\s+LISTENING\s+(\d+)/);
      if (match && match[1] !== '0') {
        execSync(`taskkill /PID ${match[1]} /F`, { timeout: 5000 });
        console.log(`[server] Freed port 5000 (killed PID ${match[1]})`);
      }
    } else {
      execSync('fuser -k 5000/tcp 2>/dev/null || true', { timeout: 5000 });
    }
  } catch {
    // port was already free or kill failed — proceed anyway
  }
}

function startPythonService() {
  const py = spawn(
    PYTHON_PATH,
    ['-m', 'uvicorn', 'src.python.main:app', '--host', '127.0.0.1', '--port', '5000', '--workers', '1'],
    {
      cwd: process.cwd(),
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    },
  );
  py.stdout.on('data', (d) => process.stdout.write(`[python] ${d}`));
  py.stderr.on('data', (d) => process.stderr.write(`[python] ${d}`));
  py.on('exit', (code) => {
    console.error(`[python] Service exited with code ${code}. Shutting down.`);
    process.exit(1);
  });
  return py;
}

async function waitForPythonReady(maxMs = 300_000) {
  const start = Date.now();
  while (Date.now() - start < maxMs) {
    try {
      await axios.get(`${PYTHON_SERVICE_URL}/health`, { timeout: 2000 });
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
  throw new Error('Python FastAPI service did not become ready within 5 minutes.');
}

(async () => {
  killPort5000();
  console.log('[server] Starting Python FastAPI service...');
  startPythonService();

  console.log('[server] Waiting for Surya OCR + opus-mt to load (takes ~1-2 min on CPU)...');
  await waitForPythonReady();
  console.log('[server] Python service ready.');

  app.listen(PORT, () => {
    console.log(`PDF Translator API running on port ${PORT}`);
  });
})().catch((err) => {
  console.error('FATAL:', err.message);
  process.exit(1);
});
