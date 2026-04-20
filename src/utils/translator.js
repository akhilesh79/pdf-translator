const axios = require('axios');

const MYMEMORY_EMAIL = process.env.MYMEMORY_EMAIL || '';
// MyMemory's free tier rejects q > 500 chars.
const CHUNK_SIZE = 480;
const MAX_CONCURRENT_REQUESTS = 2;
const REQUEST_TIMEOUT_MS = 15000;
const MAX_ATTEMPTS = 4;

if (!MYMEMORY_EMAIL) {
  console.warn(
    '[translator] MYMEMORY_EMAIL not set. Free tier limit is ~5K chars/day per IP. '
    + 'Set MYMEMORY_EMAIL in .env to raise the limit to ~50K chars/day.',
  );
}

class QuotaExceededError extends Error {
  constructor(message) {
    super(message);
    this.name = 'QuotaExceededError';
    this.code = 'QUOTA_EXCEEDED';
  }
}

// Pattern-based MyMemory error classification.
function classifyMyMemoryError(responseDetails = '') {
  const msg = String(responseDetails).toUpperCase();
  if (msg.includes('PLEASE SELECT TWO DISTINCT LANGUAGES')
    || msg.includes('SAME LANGUAGES')) {
    return 'SAME_LANGUAGE';
  }
  if (msg.includes('QUOTA')
    || msg.includes('ALL AVAILABLE FREE TRANSLATIONS')
    || msg.includes('MYMEMORY WARNING')
    || msg.includes('DAILY LIMIT')) {
    return 'QUOTA';
  }
  if (msg.includes('INVALID EMAIL')) {
    return 'INVALID_EMAIL';
  }
  if (msg.includes('AUTODETECT LIMIT')
    || msg.includes('COULD NOT')
    || msg.includes('INVALID LANGUAGE')) {
    return 'BAD_LANG';
  }
  return 'OTHER';
}

function chunkText(text) {
  const chunks = [];
  let remaining = text.trim();
  if (!remaining) return chunks;

  while (remaining.length > CHUNK_SIZE) {
    const window = remaining.slice(0, CHUNK_SIZE);
    let splitAt = -1;

    const candidates = [
      window.lastIndexOf('\n\n'),
      window.lastIndexOf('\n'),
      window.lastIndexOf('. '),
      window.lastIndexOf('? '),
      window.lastIndexOf('! '),
      window.lastIndexOf('; '),
      window.lastIndexOf(', '),
      window.lastIndexOf(' '),
    ];
    for (const c of candidates) {
      if (c >= CHUNK_SIZE * 0.5) { splitAt = c; break; }
    }
    if (splitAt <= 0) splitAt = CHUNK_SIZE;

    const piece = remaining.slice(0, splitAt).trim();
    if (piece) chunks.push(piece);
    remaining = remaining.slice(splitAt).trim();
  }

  if (remaining.length > 0) chunks.push(remaining);
  return chunks;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Returns { translatedText, passthrough: boolean }
async function translateChunkOnce(chunk, sourceLang) {
  const langPair = sourceLang === 'auto' ? 'autodetect|en' : `${sourceLang}|en`;
  const params = new URLSearchParams({ q: chunk, langpair: langPair });
  if (MYMEMORY_EMAIL) params.set('de', MYMEMORY_EMAIL);

  const url = `https://api.mymemory.translated.net/get?${params.toString()}`;
  let response;
  try {
    response = await axios.get(url, {
      timeout: REQUEST_TIMEOUT_MS,
      validateStatus: () => true,
    });
  } catch (err) {
    // Network-level failure (DNS, ECONNRESET, timeout, etc.) — retryable.
    const e = new Error(`MyMemory network error: ${err.message}`);
    e.retryable = true;
    throw e;
  }

  const result = response.data || {};
  const details = result.responseDetails || '';
  const bodyStatus = result.responseStatus;

  // Retryable transient failures.
  if (response.status === 429 || response.status >= 500
    || bodyStatus === 429 || bodyStatus >= 500) {
    const e = new Error(`MyMemory transient error ${response.status}/${bodyStatus}: ${details}`);
    e.retryable = true;
    throw e;
  }

  // Classify "client-level" errors (4xx / body != 200).
  if (response.status >= 400 || (bodyStatus && bodyStatus !== 200)) {
    const kind = classifyMyMemoryError(details);

    if (kind === 'SAME_LANGUAGE') {
      // Source already English (or detected as such). Not an error — passthrough.
      return { translatedText: chunk, passthrough: true };
    }
    if (kind === 'QUOTA') {
      throw new QuotaExceededError(
        `MyMemory daily quota exhausted. ${
          MYMEMORY_EMAIL ? '' : 'Set MYMEMORY_EMAIL in .env to raise the limit. '
        }Details: ${details}`,
      );
    }
    if (kind === 'INVALID_EMAIL') {
      throw new Error(`Configured MYMEMORY_EMAIL is invalid: ${details}`);
    }
    if (kind === 'BAD_LANG') {
      // Fall back to autodetect once instead of throwing.
      if (sourceLang !== 'auto') {
        return translateChunkOnce(chunk, 'auto');
      }
    }
    throw new Error(`MyMemory error ${response.status}/${bodyStatus}: ${details}`);
  }

  const translated = result.responseData && result.responseData.translatedText;
  if (typeof translated !== 'string') {
    throw new Error(`MyMemory returned no translatedText: ${JSON.stringify(result).slice(0, 200)}`);
  }
  return { translatedText: translated, passthrough: false };
}

async function translateChunkWithRetry(chunk, sourceLang) {
  let lastErr;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    try {
      return await translateChunkOnce(chunk, sourceLang);
    } catch (err) {
      lastErr = err;
      if (err instanceof QuotaExceededError) throw err;
      const retryable = err.retryable
        || err.code === 'ECONNABORTED'
        || err.code === 'ETIMEDOUT'
        || err.code === 'ECONNRESET';
      if (!retryable || attempt === MAX_ATTEMPTS) break;
      const delay = Math.min(1000 * 2 ** (attempt - 1), 8000);
      console.warn(`[translator] chunk attempt ${attempt} failed (${err.message}); retrying in ${delay}ms`);
      await sleep(delay);
    }
  }
  throw lastErr;
}

async function translateChunksBatch(chunks, sourceLang, concurrency = MAX_CONCURRENT_REQUESTS) {
  const results = new Array(chunks.length);
  for (let i = 0; i < chunks.length; i += concurrency) {
    const batch = chunks.slice(i, i + concurrency);
    const batchResults = await Promise.all(
      batch.map((chunk) => translateChunkWithRetry(chunk, sourceLang)),
    );
    for (let j = 0; j < batchResults.length; j += 1) {
      results[i + j] = batchResults[j];
    }
  }
  return results;
}

async function translateToEnglish(text, sourceLang = 'auto') {
  if (!text || !text.trim()) {
    return { translatedText: '', engine: 'noop' };
  }
  if (sourceLang === 'en') {
    return { translatedText: text, engine: 'passthrough' };
  }

  const chunks = chunkText(text);
  if (chunks.length === 0) {
    return { translatedText: '', engine: 'noop' };
  }

  const results = await translateChunksBatch(chunks, sourceLang);
  const translatedText = results.map((r) => r.translatedText).join('\n');
  const anyTranslated = results.some((r) => !r.passthrough);
  const anyPassthrough = results.some((r) => r.passthrough);

  let engine;
  if (anyTranslated && anyPassthrough) engine = 'mymemory+passthrough';
  else if (anyTranslated) engine = 'mymemory';
  else engine = 'passthrough';

  return { translatedText, engine };
}

module.exports = { translateToEnglish, chunkText, QuotaExceededError };
