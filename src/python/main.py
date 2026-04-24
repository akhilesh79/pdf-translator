import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, UploadFile

from .extractor import process_file
from .models import ProcessResponse
from .translator import load_nllb_model

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Surya OCR — detection + recognition (auto-multilingual, 90+ languages)
    print("[python] Loading Surya OCR models...", file=sys.stderr)
    from .surya_compat import apply as _patch_surya; _patch_surya()

    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    foundation = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor=foundation)

    models["surya"] = {"det_predictor": det_predictor, "rec_predictor": rec_predictor}
    print("[python] Surya OCR ready.", file=sys.stderr)

    # 2. NLLB-200 — local offline translation (no external API)
    models["nllb"] = load_nllb_model()

    yield
    models.clear()


app = FastAPI(title="PDF Translator Python Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "success": True,
        "surya_ready": "surya" in models,
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
        surya_models=models.get("surya"),
        nllb_model=models.get("nllb"),
    )

    if not result.get("success"):
        error = result.get("error", "Processing failed.")
        raise HTTPException(status_code=422, detail=error)

    return result
