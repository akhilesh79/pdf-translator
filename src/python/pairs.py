"""Generic label / value extractor.

Where `fields.py` knows specific PMJAY fields up front via regex, this module
walks every OCR line and extracts ANY `Label : Value` pair it finds. This
catches structured key-values the regex layer didn't anticipate (sample IDs,
ward, ref numbers, package names, lab parameters, etc.) and lets the JSON
response surface "whatever is relevant in the file".

Pairs are returned as Field objects so they merge into the same response array.
Provenance comes from the LineIndex line containing the label.
"""
from __future__ import annotations

import re
from typing import Optional

from .indexing import LineIndex
from .models import Field, Provenance


# ---------------------------------------------------------------------------
# Canonical-key normalization
# ---------------------------------------------------------------------------
# Maps lower-cased label phrasings to canonical schema keys. Anything not in
# this table keeps a snake_cased version of the original label as its key.

LABEL_TO_KEY: dict[str, str] = {
    # patient identity
    "patient name": "patient_name",
    "patients name": "patient_name",
    "patient's name": "patient_name",
    "name of patient": "patient_name",
    "name of the patient": "patient_name",
    "name": "patient_name",
    "pt name": "patient_name",
    "pt": "patient_name",
    "रोगी का नाम": "patient_name",
    "मरीज का नाम": "patient_name",
    "मरीज़ का नाम": "patient_name",
    # ids
    "uhid": "uhid",
    "u.h.i.d": "uhid",
    "u.h.i.d.": "uhid",
    "patient id": "uhid",
    "hospital id": "uhid",
    "mrn": "uhid",
    "mr no": "uhid",
    "mr. no": "uhid",
    "mr no.": "uhid",
    "ipd no": "uhid",
    "ipd no.": "uhid",
    "claim id": "claim_id",
    "claim no": "claim_id",
    "claim number": "claim_id",
    "pre auth id": "claim_id",
    "pre-auth id": "claim_id",
    "preauth id": "claim_id",
    "preauth no": "claim_id",
    "pre auth no": "claim_id",
    "tms id": "claim_id",
    "tms no": "claim_id",
    "transaction id": "claim_id",
    "reg/ref": "registration_id",
    "registration no": "registration_id",
    "reference no": "registration_id",
    # demographics
    "age": "age",
    "age/gender": "age_gender",
    "age/sex": "age_gender",
    "gender": "gender",
    "sex": "gender",
    "dob": "dob",
    "date of birth": "dob",
    "phone": "phone",
    "phone no": "phone",
    "mobile": "phone",
    "mobile no": "phone",
    "address": "address",
    # dates
    "admission date": "admission_date",
    "date of admission": "admission_date",
    "doa": "admission_date",
    "admitted on": "admission_date",
    "admit date": "admission_date",
    "discharge date": "discharge_date",
    "date of discharge": "discharge_date",
    "dod": "discharge_date",
    "discharged on": "discharge_date",
    "report date": "report_date",
    "date": "date",
    # clinical
    "diagnosis": "diagnosis",
    "final diagnosis": "diagnosis",
    "provisional diagnosis": "diagnosis",
    "dx": "diagnosis",
    "chief complaint": "chief_complaint",
    "complaint": "chief_complaint",
    "history": "history",
    "past history": "past_history",
    "personal history": "personal_history",
    "allergies": "allergies",
    "treatment given in the hospital": "treatment",
    "treatment": "treatment",
    "course during hospitalization": "course",
    "investigation": "investigation",
    "procedure": "procedure",
    "operation": "procedure",
    "surgery": "procedure",
    "advice": "advice",
    "advice on discharge": "advice",
    # facility / staff
    "hospital name": "hospital_name",
    "hospital": "hospital_name",
    "institution": "hospital_name",
    "facility": "hospital_name",
    "consultant": "doctor_name",
    "doctor": "doctor_name",
    "dr.": "doctor_name",
    "physician": "doctor_name",
    "attending": "doctor_name",
    "ref by": "referring_doctor",
    "ref.by": "referring_doctor",
    "ward": "ward",
    "bed no": "bed_no",
    "department": "department",
    # lab
    "requested test": "requested_test",
    "sample id": "sample_id",
    "coll time": "collection_time",
    "collection time": "collection_time",
    "validate": "validation_time",
    "prn. time": "print_time",
    "report status": "report_status",
    # money
    "total amount": "total_amount",
    "total cost": "total_amount",
    "grand total": "total_amount",
    "bill amount": "total_amount",
    "amount claimed": "total_amount",
    "claim amount": "total_amount",
    "total": "total_amount",
    "amount": "amount",
    "package": "package",
    "package code": "pmjay_procedure_codes",
    "package name": "package_name",
}


def _normalize_label(label: str) -> str:
    s = label.strip().lower()
    # Drop trailing colons/dashes/dots
    s = re.sub(r"[:\-\.]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_snake(label: str) -> str:
    s = _normalize_label(label)
    s = re.sub(r"[^a-z0-9ऀ-ॿ]+", "_", s)
    s = s.strip("_")
    return s or "field"


def _key_for_label(label: str) -> str:
    norm = _normalize_label(label)
    return LABEL_TO_KEY.get(norm, _to_snake(norm))


# ---------------------------------------------------------------------------
# Pair detection
# ---------------------------------------------------------------------------

# Same-line:  Label : Value
# Separator must include a colon — pure-dash separators trigger too many false
# positives mid-sentence ("Pt - Avtar Singh"). We DO accept ":-" or ":--" so
# discharge templates with "Diagnosis:- VALUE" still match.
_SAMELINE_RE = re.compile(
    r"^\s*([A-Za-zऀ-ॿ][A-Za-z0-9 .'/&ऀ-ॿ]{1,50}?)\s*:[:\-\s]{0,3}\s*(.{1,250}?)\s*$"
)
# Label only:  Label :    (value would be on next line)
_LABELONLY_RE = re.compile(
    r"^\s*([A-Za-zऀ-ॿ][A-Za-z0-9 .'/&ऀ-ॿ]{1,50}?)\s*:[:\-]{0,2}\s*$"
)

# Reject lines that look like noise / not a real label-value
_BAD_LABEL_TOKENS = {
    "page", "of", "and", "the", "a", "an", "is", "this", "that", "to", "from",
    "above", "below", "with", "without", "by", "on", "at", "in", "for", "as",
    "we", "i", "you", "they", "it", "his", "her", "our", "their", "my",
}


def _looks_like_label(text: str) -> bool:
    t = text.strip()
    if len(t) < 2 or len(t) > 60:
        return False
    # Pure digits / dates aren't labels
    if re.fullmatch(r"[\d\-/.,:\s]+", t):
        return False
    # Filler words alone aren't labels
    if t.lower() in _BAD_LABEL_TOKENS:
        return False
    # Reject very long sentences (more than 8 words)
    if len(t.split()) > 8:
        return False
    return True


def _looks_like_value(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # Reject values that are themselves clearly another label line.
    if t.endswith(":") or t.endswith(":-"):
        return False
    # Reject empty / pure-punctuation values like "-", ":-", "..." — these are
    # blank-template placeholders left in the doc, not real data.
    stripped = re.sub(r"[\s\-\.:_;,]+", "", t)
    if not stripped:
        return False
    if len(stripped) < 2:
        return False
    return True


def _line_records(line_index: LineIndex) -> list[tuple[int, list[float], str]]:
    """Reconstruct (page, bbox, text) per line from LineIndex internals."""
    out: list[tuple[int, list[float], str]] = []
    text = line_index.text
    for start, end, page, bbox in zip(
        line_index._starts, line_index._ends, line_index._pages, line_index._bboxes,
    ):
        out.append((page, bbox, text[start:end]))
    return out


def extract_pairs(line_index: LineIndex) -> list[Field]:
    """Walk every OCR line, emit Field for every plausible Label:Value pair."""
    records = _line_records(line_index)
    n = len(records)
    out: list[Field] = []

    for i, (page, bbox, line) in enumerate(records):
        line = line.strip()
        if not line:
            continue

        # Same-line pair
        m = _SAMELINE_RE.match(line)
        if m:
            label = m.group(1).strip()
            value = m.group(2).strip()
            if _looks_like_label(label) and _looks_like_value(value):
                out.append(Field(
                    key=_key_for_label(label),
                    value=value,
                    confidence=0.85,
                    source=Provenance(page=page, bbox=bbox or [0.0, 0.0, 0.0, 0.0]),
                    extractor=f"pair:{_normalize_label(label)[:40]}",
                ))
                continue

        # Label-only line: try the very next line as value (must be plausible).
        m2 = _LABELONLY_RE.match(line)
        if m2 and i + 1 < n:
            label = m2.group(1).strip()
            if not _looks_like_label(label):
                continue
            next_page, next_bbox, next_text = records[i + 1]
            next_text = next_text.strip()
            if not _looks_like_value(next_text):
                continue
            # Avoid pairing label with another label.
            if _LABELONLY_RE.match(next_text) or _SAMELINE_RE.match(next_text):
                continue
            # Same-page only — values on the next page are usually unrelated.
            if next_page != page:
                continue
            out.append(Field(
                key=_key_for_label(label),
                value=next_text,
                confidence=0.75,
                source=Provenance(page=next_page, bbox=next_bbox or [0.0, 0.0, 0.0, 0.0]),
                extractor=f"pair:{_normalize_label(label)[:40]}_nextline",
            ))

    return out
