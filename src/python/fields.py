"""Bilingual field extractor for PMJAY / medical insurance documents.

Works on the original-language OCR text (provenance source) and optionally on
the English translation (corroboration). Values agreeing across both sides get
a confidence boost.

Every Field carries (page, bbox) provenance resolved via LineIndex.
"""
from __future__ import annotations

import re
from typing import Optional

from .indexing import LineIndex
from .models import Field, Provenance


# ---------------------------------------------------------------------------
# Bilingual keyword anchors — presence near a regex match boosts confidence.
# Keys are field names; values are substrings we look for in source text.
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, list[str]] = {
    "patient_name": [
        "patient name", "name of patient", "patient's name", "name of the patient",
        "रोगी का नाम", "मरीज़ का नाम", "मरीज का नाम", "नाम",
    ],
    "uhid": ["uhid", "u.h.i.d", "hospital id", "patient id", "mrn", "mr no", "mr. no"],
    "claim_id": [
        "claim id", "claim no", "claim number", "pre auth id", "pre-auth id",
        "preauth id", "tms id", "tms no", "transaction id",
    ],
    "age": ["age", "उम्र", "आयु"],
    "gender": ["gender", "sex", "लिंग"],
    "admission_date": [
        "admission date", "date of admission", "doa", "admitted on", "admit date",
        "भर्ती की तारीख", "भर्ती दिनांक",
    ],
    "discharge_date": [
        "discharge date", "date of discharge", "dod", "discharged on",
        "डिस्चार्ज की तारीख", "डिस्चार्ज दिनांक",
    ],
    "diagnosis": [
        "diagnosis", "final diagnosis", "provisional diagnosis", "dx",
        "निदान",
    ],
    "procedure": [
        "procedure", "procedure performed", "operation", "surgery",
        "प्रक्रिया", "शल्य चिकित्सा",
    ],
    "hospital_name": [
        "hospital name", "hospital", "institution", "facility",
        "अस्पताल",
    ],
    "doctor_name": [
        "doctor", "dr.", "consultant", "physician", "attending",
        "चिकित्सक", "डॉक्टर", "डॉ.",
    ],
    "total_amount": [
        "total amount", "total cost", "grand total", "total",
        "amount claimed", "claim amount", "bill amount",
        "कुल राशि", "कुल",
    ],
}


# ---------------------------------------------------------------------------
# Regex patterns — (pattern, extractor_id, capture_group_index)
# Use re.IGNORECASE unless the pattern is script-specific.
# ---------------------------------------------------------------------------

def _c(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


# Stop-words that often follow a name and signal the name is over.
_NAME_STOP = (
    r"(?=[ \t]+(?:age|years?|yrs?|y/?o|male|female|gender|sex|adm|admitted|"
    r"hospital|h[oa]spital|m[oa]spital|date|dob|patient\s*id|uhid|preauth|"
    r"per[ei]?auth|tms|claim|pkg|package|wt|weight|son|s/o|d/o|w/o)|"
    r"[ \t]*[,;\n]|[ \t]*$)"
)
_NAME = r"([A-Z][A-Za-z.\-'']*(?:[ \t]+[A-Z][A-Za-z.\-'']*){0,4})" + _NAME_STOP
_DEVANAGARI_NAME = r"([ऀ-ॿ]+(?:[ \t]+[ऀ-ॿ]+){0,4})"
_NUM_ID = r"([A-Z0-9][A-Z0-9\-/]{4,24})"
_DATE = r"(\d{1,2}[\-/.](?:\d{1,2}|[A-Za-z]{3,9})[\-/.]\d{2,4})"
_AMOUNT = r"(?:₹|Rs\.?|INR|रु\.?)\s?([\d,]+(?:\.\d{1,2})?)"

# Pattern tuple: (compiled_regex, extractor_id, anchored)
# anchored=True means the regex itself includes a keyword prefix ("Patient Name:",
# "DOA", etc.) — the value is high-confidence. anchored=False means we matched a
# bare value (ICD code, package code) and need corroboration from context.
# OCR-error-tolerant fragments. Fuzz tolerates one substitution / insertion in
# noisy keywords ("Pereauth" / "preauth", "clingnosis" / "diagnosis", "Maspital" / "Hospital").
_PATIENT_HEAD = r"\b(?:patient(?:'s)?\s*name|name\s*of\s*(?:the\s*)?patient|pt\.?(?:\s*name)?)"
_PREAUTH = r"\b(?:pre[\s\-]?auth|per[ei]?auth|tms|claim)\s*(?:id|no\.?|number)?"
_DIAGNOSIS_HEAD = r"\b(?:final\s+diagn\w+|provisional\s+diagn\w+|diagn\w+|cling?nos\w+|d[xs]\b)"
_HOSPITAL_TAIL = r"(?:hospital|h[oa]spital|m[oa]spital|nursing\s*home|medical\s*(?:centre|center|college)|clinic|institute)"

PATTERNS: dict[str, list[tuple[re.Pattern, str, bool]]] = {
    "patient_name": [
        # Standard "Patient Name: <Name>"
        (_c(_PATIENT_HEAD + r"\s*[:\-]+[ \t]*" + _NAME),
         "regex:patient_name_en", True),
        # Handwritten / abbreviated "Pt - Name" or "Pt. Name"
        (_c(r"\bpt\.?\s*[:\-]+[ \t]*" + _NAME),
         "regex:patient_name_short", True),
        (_c(r"(?:रोगी\s*का\s*नाम|मरीज़?\s*का\s*नाम)\s*[:\-]*[ \t]*" + _DEVANAGARI_NAME),
         "regex:patient_name_hi", True),
    ],
    "uhid": [
        (_c(r"\b(?:uhid|u\.h\.i\.d\.?|hospital\s*id|patient\s*id|mrn|mr\.?\s*no\.?)\s*[:\-]*[ \t]*" + _NUM_ID),
         "regex:uhid", True),
    ],
    "claim_id": [
        (_c(_PREAUTH + r"\s*[:\-]*[ \t]*" + _NUM_ID),
         "regex:claim_id", True),
        (_c(r"\btransaction\s*id\s*[:\-]*[ \t]*" + _NUM_ID),
         "regex:transaction_id", True),
    ],
    "age": [
        (_c(r"\bage\s*(?:/\s*(?:gender|sex))?\s*[:\-]*[ \t]*(\d{1,3})\s*(?:years?|yrs?|y)?\b"),
         "regex:age_en", True),
        (_c(r"(?:उम्र|आयु)\s*[:\-]*[ \t]*(\d{1,3})"), "regex:age_hi", True),
        # Fallback: bare "<Number> Y/yrs old" (lower confidence — anchored=False)
        (_c(r"\b(\d{1,3})\s*(?:years?\s*old|y/?o|yrs?\s*old)\b"), "regex:age_bare", False),
    ],
    "gender": [
        (_c(r"\b(?:gender|sex)\s*[:\-]*[ \t]*(male|female|m|f|transgender|other)\b"), "regex:gender_en", True),
        (_c(r"लिंग\s*[:\-]*[ \t]*([ऀ-ॿ]+)"), "regex:gender_hi", True),
        # Bare gender mention near age/sex — lower confidence.
        (_c(r"\b(male|female|transgender)\b"), "regex:gender_bare", False),
    ],
    "admission_date": [
        (_c(r"\b(?:admission\s*date|date\s*of\s*admission|doa|admitted\s*on|admit\s*date)\s*[:\-]*[ \t]*" + _DATE),
         "regex:admission_date_en", True),
        (_c(r"(?:भर्ती\s*(?:की\s*)?(?:तारीख|दिनांक))\s*[:\-]*[ \t]*" + _DATE),
         "regex:admission_date_hi", True),
    ],
    "discharge_date": [
        (_c(r"\b(?:discharge\s*date|date\s*of\s*discharge|dod|discharged\s*on)\s*[:\-]*[ \t]*" + _DATE),
         "regex:discharge_date_en", True),
        (_c(r"(?:डिस्चार्ज\s*(?:की\s*)?(?:तारीख|दिनांक))\s*[:\-]*[ \t]*" + _DATE),
         "regex:discharge_date_hi", True),
    ],
    "diagnosis": [
        (_c(_DIAGNOSIS_HEAD + r"\s*(?:of\s+|[:\-]+)\s*([^\n\r]{3,200})"),
         "regex:diagnosis_en", True),
        (_c(r"निदान\s*[:\-]*[ \t]*([^\n\r]{3,200})"), "regex:diagnosis_hi", True),
    ],
    "procedure": [
        (_c(r"\b(?:procedure\s*(?:performed)?|operation|surgery)\s*[:\-]+[ \t]*([^\n\r]{3,200})"),
         "regex:procedure_en", True),
    ],
    "icd_codes": [
        (_c(r"\b([A-Z]\d{2}(?:\.\d{1,3})?)\b"), "regex:icd10", False),
    ],
    "pmjay_procedure_codes": [
        # PMJAY package code: letter + hyphen + digits (e.g. M-07-007),
        # plus generic PROC-NNNN fallback.
        (_c(r"\b([A-Z]{1,3}\-\d{2,3}\-\d{2,4}[A-Z]?)\b"), "regex:pmjay_pkg", False),
        (_c(r"\b(PROC\-\d{3,5})\b"), "regex:proc_code", False),
    ],
    "hospital_name": [
        (_c(r"\b(?:hospital\s*name|institution|facility)\s*[:\-]+[ \t]*([^\n\r]{3,120})"),
         "regex:hospital_name_en", True),
    ],
    "doctor_name": [
        (_c(r"\bdr\.?\s+" + _NAME), "regex:doctor_name", True),
    ],
    "total_amount": [
        (_c(r"\b(?:total\s*amount|total\s*cost|grand\s*total|amount\s*claimed|claim\s*amount|bill\s*amount|total)\s*[:\-]*[ \t]*" + _AMOUNT),
         "regex:total_amount_en", True),
        (_c(r"कुल\s*(?:राशि)?\s*[:\-]*[ \t]*" + _AMOUNT), "regex:total_amount_hi", True),
    ],
}

REQUIRED_FIELDS = {"patient_name", "uhid", "diagnosis", "admission_date", "discharge_date"}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _normalize_value(field: str, value: str) -> str:
    value = value.strip().strip(".,;:")
    value = re.sub(r"\s+", " ", value)
    if field in {"patient_name", "doctor_name", "hospital_name"}:
        # Title-case simple Latin names; leave Devanagari untouched.
        if re.fullmatch(r"[A-Za-z.\-' ]+", value):
            value = " ".join(w.capitalize() for w in value.split())
    if field == "total_amount":
        value = value.replace(",", "")
    if field == "gender":
        first = value.strip().lower()[:1]
        if first == "m":
            value = "Male"
        elif first == "f":
            value = "Female"
    return value


def _extract_side(text: str, line_index: Optional[LineIndex], lang: str) -> list[Field]:
    """Run all patterns against one text side. lang is 'src' or 'en' — used in extractor id."""
    out: list[Field] = []
    for field_key, pattern_list in PATTERNS.items():
        for pattern, extractor_id, anchored in pattern_list:
            for m in pattern.finditer(text):
                value = m.group(1) if m.groups() else m.group(0)
                if not value:
                    continue
                value = _normalize_value(field_key, value)
                if not value:
                    continue
                # Anchored = regex baked the keyword in; high confidence.
                # Bare = generic value pattern; lower confidence.
                confidence = 1.0 if anchored else 0.7
                if lang == "en" and field_key not in {"icd_codes", "pmjay_procedure_codes"}:
                    # Translated-side evidence is weaker (possible hallucination).
                    confidence = min(confidence, 0.5)

                source: Optional[Provenance] = None
                if lang == "src" and line_index is not None:
                    source = line_index.lookup_span(m.start(), m.end())

                out.append(Field(
                    key=field_key,
                    value=value,
                    confidence=confidence,
                    source=source,
                    extractor=f"{extractor_id}:{lang}",
                ))
    return out


def _dedup_and_merge(src_fields: list[Field], en_fields: list[Field]) -> list[Field]:
    """For fields where a single value is expected, keep the highest-confidence
    entry. For multi-value fields (icd_codes, procedures), keep all unique
    values. When the same normalized value appears on both sides, add an
    agreement bonus (+0.1, capped at 1.0).
    """
    SINGLE_VALUE = {
        "patient_name", "uhid", "claim_id", "age", "gender",
        "admission_date", "discharge_date", "hospital_name", "total_amount",
    }

    # Group by (field_key, normalized_value_lower)
    combined = src_fields + en_fields
    # Agreement bonus
    by_key_val: dict[tuple[str, str], list[Field]] = {}
    for f in combined:
        key = (f.key, f.value.lower())
        by_key_val.setdefault(key, []).append(f)

    merged: list[Field] = []
    for (key, _v), fs in by_key_val.items():
        if len(fs) > 1:
            # Agreement bonus — take best source and boost confidence.
            best = max(fs, key=lambda x: x.confidence)
            boosted = Field(
                key=best.key,
                value=best.value,
                confidence=min(1.0, best.confidence + 0.1),
                source=best.source,
                extractor=best.extractor,
            )
            merged.append(boosted)
        else:
            merged.append(fs[0])

    # For single-value fields, keep only the top candidate.
    by_key: dict[str, list[Field]] = {}
    for f in merged:
        by_key.setdefault(f.key, []).append(f)

    final: list[Field] = []
    for key, fs in by_key.items():
        if key in SINGLE_VALUE:
            final.append(max(fs, key=lambda x: (x.confidence, len(x.value))))
        else:
            # Multi-value: dedupe by value (case-insensitive), keep highest conf each.
            seen: dict[str, Field] = {}
            for f in fs:
                v = f.value.lower()
                if v not in seen or f.confidence > seen[v].confidence:
                    seen[v] = f
            final.extend(seen.values())

    return final


def extract_fields(
    source_text: str,
    line_index: LineIndex,
    translated_text: Optional[str] = None,
) -> list[Field]:
    """Main entry point. Runs extraction on source text (with provenance) and
    optionally on translated text (corroboration), then merges."""
    src_fields = _extract_side(source_text, line_index, "src")

    en_fields: list[Field] = []
    if translated_text and translated_text.strip() and translated_text != source_text:
        en_fields = _extract_side(translated_text, None, "en")

    return _dedup_and_merge(src_fields, en_fields)
