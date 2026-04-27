from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


# ── Internal collection models (used during processing) ──────────────────────

class Provenance(BaseModel):
    page: int
    bbox: list[float]


class Field(BaseModel):
    key: str
    value: str
    confidence: float
    source: Optional[Provenance] = None
    extractor: str


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
    title: Optional[str] = None


class VisualElement(BaseModel):
    type: str
    page: int
    bbox: list[float]
    data: Optional[str] = None
    confidence: float


class TimelineEvent(BaseModel):
    date: str
    event: str
    description: Optional[str] = None
    source: Optional[Provenance] = None


# ── Output response models ────────────────────────────────────────────────────

class Metadata(BaseModel):
    language: Optional[str] = None
    pages: Optional[int] = None
    source_type: str = "pdf"
    is_scanned: bool = False
    processing_time_ms: Optional[int] = None


class OcrPage(BaseModel):
    page_number: int
    text: str
    confidence: Optional[float] = None


class OcrBlock(BaseModel):
    engine: str
    avg_confidence: Optional[float] = None
    original_text: str
    translated_text: Optional[str] = None


class LayoutBlock(BaseModel):
    sections: list = []
    tables: list[Table] = []
    line_items: list = []


class VisualElementsBlock(BaseModel):
    stamps: list[VisualElement] = []
    signatures: list[VisualElement] = []
    qr_codes: list[VisualElement] = []
    barcodes: list[VisualElement] = []
    implant_stickers: list = []


class RuleCheck(BaseModel):
    rule_id: str
    description: str
    status: str  # "pass" | "fail" | "conditional"
    reason: Optional[str] = None


class RulesEngine(BaseModel):
    checks: list[RuleCheck] = []


class EvidenceItem(BaseModel):
    field: str
    value: str
    page: Optional[int] = None
    bbox: Optional[list[float]] = None


class Explainability(BaseModel):
    confidence_score: float
    evidence: list[EvidenceItem] = []


class FlagItem(BaseModel):
    type: str
    priority: str  # "high" | "medium" | "low"
    message: str


class DecisionBlock(BaseModel):
    status: str
    reason: Optional[str] = None
    recommended_action: Optional[str] = None


class ProcessResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    document_id: Optional[str] = None
    metadata: Optional[Metadata] = None
    ocr: Optional[OcrBlock] = None
    layout: Optional[LayoutBlock] = None
    extracted_fields: Optional[dict] = None
    visual_elements: Optional[VisualElementsBlock] = None
    rules_engine: Optional[RulesEngine] = None
    timeline: list[TimelineEvent] = []
    explainability: Optional[Explainability] = None
    flags: list[FlagItem] = []
    decision: Optional[DecisionBlock] = None
