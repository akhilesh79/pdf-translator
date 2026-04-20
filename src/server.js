require('dotenv').config();
const express = require('express');
const translateRouter = require('./routes/translate');
const { verifyPythonDeps } = require('./utils/pdfExtractorPython');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '50mb' }));
app.use('/api/translate', translateRouter);

app.get('/health', (req, res) => {
  res.json({ success: true, message: 'Server Health is Awesome!' });
});

app.use((req, res) => {
  res.status(404).json({ success: false, error: 'Route not found' });
});

app.use((err, req, res, next) => {
  console.error('Handling error:', err.message);
  return res.status(500).json({ success: false, error: err.message });
});

verifyPythonDeps()
  .then(() => {
    console.log('[server] Python dependencies OK');
    app.listen(PORT, () => {
      console.log(`PDF Translator API running on port ${PORT}`);
    });
  })
  .catch((err) => {
    console.error('FATAL: ' + err.message);
    process.exit(1);
  });
