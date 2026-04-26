import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, UploadFile

# Surya CPU batch sizes. On RAM-constrained boxes, going higher than the
# defaults (det=8, rec=32) causes paging which dominates inference time.
# Empirically the defaults are best on this 6 GB Windows machine.
os.environ.setdefault("DETECTOR_BATCH_SIZE", "8")
os.environ.setdefault("RECOGNITION_BATCH_SIZE", "32")

# OCR speed tuning — recognition is the dominant cost (~99% of time on CPU).
# These knobs reduce per-region work without removing meaningful text:
#  - DETECTOR_TEXT_THRESHOLD bumped 0.6 → 0.65: fewer very low-confidence
#    detection boxes get fed to recognition. Filters OCR noise on dirty scans.
#  - FOUNDATION_MAX_TOKENS = 96: cap recognition decoder output length.
#    Most lines decode in 30-60 tokens; capping at 96 stops the model from
#    chasing long generations on noisy regions. Real-line truncation is rare.
os.environ.setdefault("DETECTOR_TEXT_THRESHOLD", "0.65")
os.environ.setdefault("FOUNDATION_MAX_TOKENS", "96")

# Cap thread count to PHYSICAL cores. Modern x86 CPUs share SIMD/FP units
# between hyperthread pairs; oversubscribing usually hurts torch matmul.
# `os.cpu_count()` returns logical (12 on a 6-core SMT CPU); halving gives
# physical. This matters more for OCR throughput than any batch tuning.
import multiprocessing as _mp
_phys = max(1, (_mp.cpu_count() or 2) // 2)
os.environ.setdefault("OMP_NUM_THREADS", str(_phys))
os.environ.setdefault("MKL_NUM_THREADS", str(_phys))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_phys))

from .extractor import process_file
from .models import ProcessResponse
from .translator import load_nllb_model

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Surya OCR — detection + recognition (auto-multilingual, 90+ languages)
    print(f"[python] Surya batch sizes: det={os.environ['DETECTOR_BATCH_SIZE']} "
          f"rec={os.environ['RECOGNITION_BATCH_SIZE']}", file=sys.stderr)
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
