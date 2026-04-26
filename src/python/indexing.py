"""Line-level character offset index: maps offsets in concatenated cleaned text
back to the original (page, bbox) source. Used by fields, timeline, and decision
modules to attach provenance to every extracted value.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Optional

from .models import Provenance


@dataclass
class LineIndex:
    text: str = ""
    _starts: list[int] = field(default_factory=list)
    _ends: list[int] = field(default_factory=list)
    _pages: list[int] = field(default_factory=list)
    _bboxes: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_cleaned_lines(
        cls, lines: list[tuple[int, list[float], str]]
    ) -> "LineIndex":
        """Build the index from pre-cleaned lines.

        Each tuple is (page_number_1_based, bbox_or_empty, text). Pages are
        separated by a blank line in the concatenated text so downstream
        chunkers see paragraph boundaries.
        """
        idx = cls()
        parts: list[str] = []
        cursor = 0
        last_page: Optional[int] = None

        for page, bbox, text in lines:
            if last_page is not None and page != last_page:
                parts.append("")           # page separator (blank line)
                cursor += 1                # for the extra \n
            parts.append(text)
            idx._starts.append(cursor)
            idx._ends.append(cursor + len(text))
            idx._pages.append(page)
            idx._bboxes.append(list(bbox) if bbox else [])
            cursor += len(text) + 1        # +1 for \n separator
            last_page = page

        idx.text = "\n".join(parts)
        return idx

    def lookup(self, char_offset: int) -> Optional[Provenance]:
        """Return the line that contains `char_offset`, or the nearest one."""
        if not self._starts:
            return None
        i = bisect.bisect_right(self._starts, char_offset) - 1
        if i < 0:
            i = 0
        bbox = self._bboxes[i] if self._bboxes[i] else [0.0, 0.0, 0.0, 0.0]
        return Provenance(page=self._pages[i], bbox=bbox)

    def lookup_span(self, start: int, end: int) -> Optional[Provenance]:
        return self.lookup((start + end) // 2)

    def __len__(self) -> int:
        return len(self._starts)
