# ── Stage 1: Python deps + EasyOCR model pre-bake ────────────────────
# Isolated from the runtime so build tools (gcc, etc.) don't bloat the
# final image. EasyOCR (~200 MB) is baked here so containers start cold.
FROM python:3.11-slim-bookworm AS py-deps

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgl1-mesa-glx \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv — copied verbatim into the runtime stage.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake EasyOCR model weights so the container doesn't download them
# on every cold start. Models are written to /root/.EasyOCR (~200 MB).
RUN python3 -c "import easyocr; easyocr.Reader(['hi', 'en'], gpu=False, verbose=False)"


# ── Stage 2: Node prod deps ───────────────────────────────────────────
FROM node:20-slim AS node-deps

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev


# ── Stage 3: Runtime image ────────────────────────────────────────────
# python:3.11-slim-bookworm base keeps the venv paths identical to Stage 1.
# Node.js is added via NodeSource so both runtimes live in one image.
FROM python:3.11-slim-bookworm

# Runtime system libs + Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      poppler-utils \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python venv (all packages + uvicorn binary) from Stage 1
COPY --from=py-deps /opt/venv /opt/venv

# Pre-baked EasyOCR model weights from Stage 1
COPY --from=py-deps /root/.EasyOCR /root/.EasyOCR

# Production Node modules from Stage 2
COPY --from=node-deps /app/node_modules ./node_modules

# Application source (respects .dockerignore)
COPY . .

# /opt/venv/bin on PATH → `python3 -m uvicorn` resolves to the venv.
# HF_HOME → Render persistent disk (mount at /app/.hf_cache) so NLLB-200
# survives redeploys without re-downloading 1.2 GB each time.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHON_PATH="/opt/venv/bin/python3" \
    HF_HOME="/app/.hf_cache" \
    HF_HUB_DISABLE_SYMLINKS_WARNING="1"

EXPOSE 3000

# Node spawns uvicorn (FastAPI) as a child process, waits for /health,
# then begins accepting Express requests.
CMD ["node", "src/server.js"]
