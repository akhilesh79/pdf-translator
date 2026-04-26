"""Rule-based Pass / Conditional / Fail verdict for a claim document.

Flags are human-readable reason strings — the "why" judges see.
"""
from __future__ import annotations

from datetime import datetime

from .models import Field, Table, VisualElement, TimelineEvent


REQUIRED_FIELDS = {"patient_name", "uhid", "diagnosis", "admission_date", "discharge_date"}
MIN_REQUIRED_CONFIDENCE = 0.7


def _fields_by_key(fields: list[Field]) -> dict[str, list[Field]]:
    out: dict[str, list[Field]] = {}
    for f in fields:
        out.setdefault(f.key, []).append(f)
    return out


def _best_confidence(fields: list[Field]) -> float:
    return max((f.confidence for f in fields), default=0.0)


def _best_value(fields: list[Field]) -> str:
    if not fields:
        return ""
    return max(fields, key=lambda f: f.confidence).value


def _parse_date(value: str) -> datetime | None:
    # Accept several incoming formats the field extractor might produce.
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                "%d-%b-%Y", "%d %B %Y", "%d-%B-%Y",
                "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def decide(
    fields: list[Field],
    tables: list[Table],
    visual_elements: list[VisualElement],
    timeline: list[TimelineEvent],
) -> tuple[str, list[str]]:
    flags: list[str] = []
    by_key = _fields_by_key(fields)

    # 1) Required fields must be present with enough confidence.
    hard_missing: list[str] = []
    weak: list[str] = []
    for req in REQUIRED_FIELDS:
        fs = by_key.get(req, [])
        if not fs:
            hard_missing.append(req)
        elif _best_confidence(fs) < MIN_REQUIRED_CONFIDENCE:
            weak.append(req)

    for k in hard_missing:
        flags.append(f"Missing required field: {k}")
    for k in weak:
        flags.append(f"Low-confidence required field: {k}")

    # 2) Chronology — discharge cannot precede admission.
    adm_str = _best_value(by_key.get("admission_date", []))
    dis_str = _best_value(by_key.get("discharge_date", []))
    adm_dt = _parse_date(adm_str) if adm_str else None
    dis_dt = _parse_date(dis_str) if dis_str else None
    if adm_dt and dis_dt and dis_dt < adm_dt:
        flags.append(f"Chronology violation: discharge ({dis_str}) before admission ({adm_str})")

    # 3) Timeline consistency — no discharge event should predate any admission event.
    adm_timeline = sorted(e.date for e in timeline if e.event == "admission")
    dis_timeline = sorted(e.date for e in timeline if e.event == "discharge")
    if adm_timeline and dis_timeline and dis_timeline[0] < adm_timeline[0]:
        flags.append(
            f"Timeline inconsistency: earliest discharge ({dis_timeline[0]}) before "
            f"earliest admission ({adm_timeline[0]})"
        )

    # 4) Visual authentication cues.
    has_stamp = any(v.type == "stamp" for v in visual_elements)
    has_signature = any(v.type == "signature" for v in visual_elements)
    if not has_stamp:
        flags.append("No hospital stamp detected")
    if not has_signature:
        flags.append("No signature detected")

    # 5) Verdict calculation.
    if hard_missing:
        return "FAIL", flags
    if adm_dt and dis_dt and dis_dt < adm_dt:
        return "FAIL", flags
    if weak or not has_stamp or not has_signature:
        return "CONDITIONAL", flags
    return "PASS", flags
