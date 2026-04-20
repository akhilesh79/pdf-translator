# ---------------------------------------------------------------
# PDF Translator — runs Node (Express) + Python (pdfplumber/OCR)
# inside one container so deploys are reproducible on Render.
# ---------------------------------------------------------------
FROM node:20-bookworm-slim

# System deps: Python, Poppler (for pdf2image), Tesseract + language packs.
# Each tesseract-ocr-<lang> adds ~5-20 MB. Trim this list if you want a smaller image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      poppler-utils \
      tesseract-ocr \
      tesseract-ocr-osd \
      tesseract-ocr-eng \
      tesseract-ocr-hin \
      tesseract-ocr-ara \
      tesseract-ocr-chi-sim \
      tesseract-ocr-chi-tra \
      tesseract-ocr-jpn \
      tesseract-ocr-kor \
      tesseract-ocr-rus \
      tesseract-ocr-fra \
      tesseract-ocr-deu \
      tesseract-ocr-spa \
      tesseract-ocr-por \
      tesseract-ocr-ita \
      tesseract-ocr-nld \
      tesseract-ocr-pol \
      tesseract-ocr-tur \
      tesseract-ocr-vie \
      tesseract-ocr-tha \
      tesseract-ocr-ell \
      tesseract-ocr-heb \
      tesseract-ocr-fas \
      tesseract-ocr-urd \
      tesseract-ocr-ben \
      tesseract-ocr-tam \
      tesseract-ocr-tel \
      tesseract-ocr-guj \
      tesseract-ocr-pan \
      tesseract-ocr-mar \
      tesseract-ocr-mal \
      tesseract-ocr-kan \
      tesseract-ocr-ukr \
  && rm -rf /var/lib/apt/lists/*

# Where tessdata lives on Debian bookworm with Tesseract 5.x.
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

WORKDIR /app

# 1. Python deps (cached layer — changes rarely).
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# 2. Node deps (cached layer).
COPY package*.json ./
RUN npm ci --omit=dev

# 3. App source.
COPY . .

# Render injects PORT at runtime; server.js already reads process.env.PORT.
EXPOSE 3000

CMD ["node", "src/server.js"]
