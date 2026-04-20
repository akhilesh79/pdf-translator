const { franc } = require('franc');

const ISO3_TO_ISO1 = {
  eng: 'en',
  fra: 'fr',
  deu: 'de',
  spa: 'es',
  por: 'pt',
  rus: 'ru',
  ara: 'ar',
  zho: 'zh',
  cmn: 'zh',
  hin: 'hi',
  jpn: 'ja',
  kor: 'ko',
  ita: 'it',
  nld: 'nl',
  pol: 'pl',
  tur: 'tr',
  swe: 'sv',
  dan: 'da',
  fin: 'fi',
  nor: 'nb',
  ukr: 'uk',
  ces: 'cs',
  ron: 'ro',
  hun: 'hu',
  vie: 'vi',
  ind: 'id',
  tha: 'th',
  ben: 'bn',
  fas: 'fa',
  heb: 'he',
  ell: 'el',
  bul: 'bg',
  hrv: 'hr',
  srp: 'sr',
  slk: 'sk',
  slv: 'sl',
  lit: 'lt',
  lav: 'lv',
  est: 'et',
  mar: 'mr',
  tam: 'ta',
  tel: 'te',
  guj: 'gu',
  pan: 'pa',
  urd: 'ur',
  msa: 'ms',
};

// Tesseract uses its own 3-letter codes. Map ISO3 (from franc) -> Tesseract.
// Defaults to 'eng' if unknown so OCR still produces something.
const ISO3_TO_TESSERACT = {
  eng: 'eng',
  fra: 'fra',
  deu: 'deu',
  spa: 'spa',
  por: 'por',
  rus: 'rus',
  ara: 'ara',
  zho: 'chi_sim',
  cmn: 'chi_sim',
  hin: 'hin',
  jpn: 'jpn',
  kor: 'kor',
  ita: 'ita',
  nld: 'nld',
  pol: 'pol',
  tur: 'tur',
  swe: 'swe',
  dan: 'dan',
  fin: 'fin',
  nor: 'nor',
  ukr: 'ukr',
  ces: 'ces',
  ron: 'ron',
  hun: 'hun',
  vie: 'vie',
  ind: 'ind',
  tha: 'tha',
  ben: 'ben',
  fas: 'fas',
  heb: 'heb',
  ell: 'ell',
  bul: 'bul',
  hrv: 'hrv',
  srp: 'srp',
  slk: 'slk',
  slv: 'slv',
  lit: 'lit',
  lav: 'lav',
  est: 'est',
  mar: 'mar',
  tam: 'tam',
  tel: 'tel',
  guj: 'guj',
  pan: 'pan',
  urd: 'urd',
  msa: 'msa',
};

function detectIso3(text) {
  if (!text || text.trim().length < 20) return null;
  const sample = text.slice(0, 4000);
  const iso3 = franc(sample, { minLength: 3 });
  return iso3 === 'und' ? null : iso3;
}

// Cheap heuristic: if >92% of chars are ASCII printable and the text contains
// several common English function words, treat it as English. Used as a
// fallback when franc returns 'und' (often on short/mixed text with digits).
const COMMON_EN_WORDS = new Set([
  'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'it',
  'with', 'as', 'was', 'on', 'are', 'be', 'this', 'by', 'from', 'or',
  'an', 'but', 'not', 'have', 'has', 'had', 'you', 'we', 'they', 'he', 'she',
]);

function looksLikeEnglish(text) {
  if (!text) return false;
  const sample = text.slice(0, 4000);
  let asciiPrintable = 0;
  for (let i = 0; i < sample.length; i += 1) {
    const c = sample.charCodeAt(i);
    if ((c >= 32 && c <= 126) || c === 9 || c === 10 || c === 13) asciiPrintable += 1;
  }
  if (asciiPrintable / sample.length < 0.92) return false;

  const tokens = sample.toLowerCase().match(/[a-z]+/g) || [];
  if (tokens.length < 10) return false;
  let hits = 0;
  for (const t of tokens) {
    if (COMMON_EN_WORDS.has(t)) hits += 1;
    if (hits >= 3) return true;
  }
  return false;
}

function detectLanguage(text) {
  const iso3 = detectIso3(text);
  if (iso3) return ISO3_TO_ISO1[iso3] || 'auto';
  // franc was undetermined — heuristic English check before falling back to auto.
  if (looksLikeEnglish(text)) return 'en';
  return 'auto';
}

function detectTesseractLang(text, fallback = 'eng') {
  const iso3 = detectIso3(text);
  if (!iso3) return fallback;
  return ISO3_TO_TESSERACT[iso3] || fallback;
}

module.exports = { detectLanguage, detectTesseractLang, looksLikeEnglish };
