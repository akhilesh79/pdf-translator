from pydantic import BaseModel
from typing import Optional


class Provenance(BaseModel):
    page: int
    bbox: list[float]  # [x0, y0, x1, y1] in image coordinates


class Field(BaseModel):
    key: str                       # patient_name, uhid, diagnosis, ...
    value: str
    confidence: float              # 0.0 - 1.0
    source: Optional[Provenance] = None
    extractor: str                 # e.g. "regex:patient_name_v2"


class TableCell(BaseModel):
    row: int
    col: int
    text: str


class Table(BaseModel):
    page: int
    bbox: Optional[list[float]] = None
    rows: int
    cols: int
    cells: list[TableCell] = []
    title: Optional[str] = None    # e.g. "Complete Blood Count"


class VisualElement(BaseModel):
    type: str                      # qr | barcode | stamp | signature
    page: int
    bbox: list[float]
    data: Optional[str] = None     # payload for QR/barcode
    confidence: float


class TimelineEvent(BaseModel):
    date: str                      # ISO YYYY-MM-DD
    event: str                     # admission | procedure | discharge | test | note
    description: Optional[str] = None
    source: Optional[Provenance] = None


class ProcessResponse(BaseModel):
    success: bool
    sourceLanguage: Optional[str] = None
    isScanned: Optional[bool] = None
    ocrLang: Optional[str] = None
    pageCount: Optional[int] = None
    engine: Optional[str] = None
    originalText: Optional[str] = None
    translatedText: Optional[str] = None
    error: Optional[str] = None

    # Structured extraction
    fields: list[Field] = []
    tables: list[Table] = []
    visual_elements: list[VisualElement] = []
    timeline: list[TimelineEvent] = []
    decision: Optional[str] = None       # PASS | CONDITIONAL | FAIL
    flags: list[str] = []
    timings: Optional[dict] = None       # per-stage milliseconds, debug aid
