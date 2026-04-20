const { LRUCache } = require('lru-cache');
const crypto = require('crypto');

const MAX_ITEMS = parseInt(process.env.CACHE_MAX_ITEMS || '100', 10);
const TTL_MS = parseInt(process.env.CACHE_TTL_MS || String(1000 * 60 * 60 * 24), 10);

const cache = new LRUCache({ max: MAX_ITEMS, ttl: TTL_MS });

function hashBuffer(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function get(key) {
  return cache.get(key);
}

function set(key, value) {
  cache.set(key, value);
}

function size() {
  return cache.size;
}

module.exports = { hashBuffer, get, set, size };
