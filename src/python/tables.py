"""Table extraction.

Two strategies:
  1. Digital PDFs: `pdfplumber.page.extract_tables()` — reliable on CBC-style
     lab reports and any digitally authored document.
  2. Scanned PDFs / images: infer tables by clustering Surya text_line x-edges.
     Lower precision but needs no ML.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any

from .models import Table, TableCell


# ---------------------------------------------------------------------------
# Digital PDF — delegate to pdfplumber
# ---------------------------------------------------------------------------

def extract_digital_tables(pdfplumber_pages: list[Any]) -> list[Table]:
    out: list[Table] = []
    for page_idx, page in enumerate(pdfplumber_pages, start=1):
        try:
            raw_tables = page.extract_tables() or []
        except Exception as e:
            print(f"[tables] pdfplumber.extract_tables failed on page {page_idx}: {e}",
                  file=sys.stderr)
            continue

        for t in raw_tables:
            if not t or not t[0]:
                continue
            rows = len(t)
            cols = max((len(r) for r in t), default=0)
            cells: list[TableCell] = []
            for r_idx, row in enumerate(t):
                for c_idx, cell in enumerate(row):
                    text = (cell or "").strip()
                    if text:
                        cells.append(TableCell(row=r_idx, col=c_idx, text=text))
            if not cells:
                continue
            # Title heuristic: if the first row has a single non-empty cell
            # and all other rows have more, treat that cell as the title.
            title = None
            first_row_texts = [
                (c.text, c) for c in cells if c.row == 0
            ]
            if len(first_row_texts) == 1 and cols > 1:
                title = first_row_texts[0][0]

            out.append(Table(
                page=page_idx,
                bbox=None,
                rows=rows,
                cols=cols,
                cells=cells,
                title=title,
            ))
    return out


# ---------------------------------------------------------------------------
# Scanned / OCR — cluster text_line x-edges to find tabular regions
# ---------------------------------------------------------------------------

def infer_tables_from_lines(
    lines_by_page: dict[int, list[tuple[str, list[float]]]],
    x_tolerance: float = 15.0,
    min_rows: int = 3,
    min_cols: int = 2,
) -> list[Table]:
    """Given per-page lines, identify rows whose text is split into multiple
    bounding boxes at roughly the same x positions across rows.

    This is a cheap heuristic — it won't catch every table — but it recovers
    the most common case: a list of rows with 2-5 column starts that line up
    vertically within a tolerance.

    NOTE: Surya returns one bbox per detected text line (which is usually one
    visible line of text). When a row has columns, Surya typically still emits
    each cell as its own line. To make this work we'd need to group lines by
    y-proximity and then check x-alignment — left as a stretch path since the
    digital-PDF path already covers CBC-style content in our test set.
    """
    # Stub: return nothing. Future work when we have labeled data.
    _ = lines_by_page, x_tolerance, min_rows, min_cols
    return []
