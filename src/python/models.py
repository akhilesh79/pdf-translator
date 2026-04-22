from pydantic import BaseModel
from typing import Optional


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
