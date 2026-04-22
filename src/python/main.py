import sys
from contextlib import asynccontextmanager

import easyocr
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .extractor import process_file
from .models import ProcessResponse
from .translator import load_nllb_model

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. EasyOCR — text extraction from scanned images
    print("[python] Loading EasyOCR (hi, en)...", file=sys.stderr)
    models["ocr"] = easyocr.Reader(["hi", "en"], gpu=False, verbose=False)
    print("[python] EasyOCR ready.", file=sys.stderr)

    # 2. NLLB-200 — local offline translation (no external API)
    models["nllb"] = load_nllb_model()

    yield
    models.clear()


app = FastAPI(title="PDF Translator Python Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "success": True,
        "easyocr_ready": "ocr" in models,
        "nllb_ready": "nllb" in models,
    }


@app.post("/process", response_model=ProcessResponse)
async def process(
    file: UploadFile,
    lang: str = Form(default="auto"),
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = process_file(
        file_bytes=file_bytes,
        filename=file.filename,
        content_type=file.content_type or "",
        lang_arg=lang,
        reader=models.get("ocr"),
        nllb_model=models.get("nllb"),
    )

    if not result.get("success"):
        error = result.get("error", "Processing failed.")
        raise HTTPException(status_code=422, detail=error)

    return result
