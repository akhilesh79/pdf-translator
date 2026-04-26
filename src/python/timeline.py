"""Episode timeline extractor.

Scans text for date mentions, pairs each with the nearest event keyword within
a ±120 char window, and returns a chronologically sorted list of events with
provenance.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from .indexing import LineIndex
from .models import TimelineEvent


# ---------------------------------------------------------------------------
# Date patterns + normalisation
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_DATE_PATTERNS = [
    # 12/04/2025, 12-04-2025, 12.04.2025
    re.compile(r"\b(\d{1,2})[\-/.](\d{1,2})[\-/.](\d{2,4})\b"),
    # 12-Apr-2025, 12 April 2025
    re.compile(r"\b(\d{1,2})[\-\s]([A-Za-z]{3,9})[\-\s](\d{2,4})\b"),
    # 2025-04-12 (ISO)
    re.compile(r"\b(\d{4})[\-/.](\d{1,2})[\-/.](\d{1,2})\b"),
]

_EVENT_KEYWORDS: dict[str, list[str]] = {
    "admission": [
        "admitted", "admission", "doa", "date of admission", "admit",
        "भर्ती",
    ],
    "discharge": [
        "discharged", "discharge", "dod", "date of discharge",
        "डिस्चार्ज",
    ],
    "procedure": [
        "surgery", "operation", "procedure", "operated", "performed",
        "शल्य", "प्रक्रिया",
    ],
    "test": [
        "test", "reported", "lab", "investigation", "sample collected",
        "जांच", "परीक्षण",
    ],
    "consultation": [
        "consulted", "visit", "opd", "follow up", "follow-up", "review",
    ],
}


def _to_iso(match: re.Match, pattern_idx: int) -> Optional[str]:
    """Normalize a date match into ISO YYYY-MM-DD. Returns None if unparseable."""
    try:
        if pattern_idx == 0:
            d, m, y = match.group(1), match.group(2), match.group(3)
            day, mon = int(d), int(m)
            year = int(y)
            if year < 100:
                year += 2000 if year < 50 else 1900
        elif pattern_idx == 1:
            d, m, y = match.group(1), match.group(2), match.group(3)
            day = int(d)
            mon = _MONTHS.get(m.lower())
            if mon is None:
                return None
            year = int(y)
            if year < 100:
                year += 2000 if year < 50 else 1900
        elif pattern_idx == 2:
            year, m, d = match.group(1), match.group(2), match.group(3)
            day, mon, year = int(d), int(m), int(year)
        else:
            return None
        dt = datetime(year, mon, day)
        # Sanity: reject dates before 1900 or more than 5 years in future.
        if dt.year < 1900 or dt.year > datetime.now().year + 5:
            return None
        return dt.strftime("%Y-%m-%d")
    except (ValueError, KeyError):
        return None


def _classify_event(text: str, match_start: int, match_end: int, window: int = 120) -> Optional[str]:
    """Pick the event whose keyword most plausibly describes this date.

    Strategy: PMJAY templates are nearly always "X Date: DD/MM/YYYY" on a
    single line — the keyword must come BEFORE the date on the same line for
    a confident classification. If same-line left-context doesn't match, fall
    back to nearest keyword within the broader window (with a left-bias).
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    left_context = text[line_start:match_start].lower()
    for event, keywords in _EVENT_KEYWORDS.items():
        for kw in keywords:
            if kw in left_context:
                return event

    # Fall back to nearest keyword within window, preferring left side.
    lo = max(0, match_start - window)
    hi = min(len(text), match_end + window // 2)
    haystack = text[lo:hi].lower()
    best_event = None
    best_dist = window + 1
    for event, keywords in _EVENT_KEYWORDS.items():
        for kw in keywords:
            idx = haystack.find(kw)
            if idx == -1:
                continue
            absolute_idx = lo + idx
            # Penalize keywords that come AFTER the date — left side wins ties.
            dist = abs(absolute_idx - match_start)
            if absolute_idx > match_start:
                dist += 30
            if dist < best_dist:
                best_dist = dist
                best_event = event
    return best_event


def _description_snippet(text: str, match_start: int, match_end: int, max_len: int = 140) -> str:
    """Pull a small context snippet — start from the current line's beginning
    so the keyword appears in the snippet."""
    line_start = text.rfind("\n", 0, match_start) + 1
    hi = min(len(text), match_end + 80)
    snippet = text[line_start:hi].replace("\n", " | ").strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rstrip() + "…"
    return snippet


def build_timeline(full_text: str, line_index: LineIndex) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    seen: set[tuple[str, str]] = set()

    for i, pattern in enumerate(_DATE_PATTERNS):
        for m in pattern.finditer(full_text):
            iso = _to_iso(m, i)
            if not iso:
                continue
            event_type = _classify_event(full_text, m.start(), m.end())
            if event_type is None:
                # No nearby event keyword — skip bare dates (letterheads, phone dates, etc.)
                continue

            key = (iso, event_type)
            if key in seen:
                continue
            seen.add(key)

            description = _description_snippet(full_text, m.start(), m.end())
            source = line_index.lookup_span(m.start(), m.end())
            events.append(TimelineEvent(
                date=iso,
                event=event_type,
                description=description,
                source=source,
            ))

    events.sort(key=lambda e: e.date)
    return events
