"""Visual cue detection — runs on rasterized PIL page images.

All detectors use classical computer vision (pyzbar + OpenCV); no ML models
are loaded. Each returns a list[VisualElement] with page number (1-based)
and pixel-space bbox [x0, y0, x1, y1].
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from .models import VisualElement


# ---------------------------------------------------------------------------
# QR codes + 1-D barcodes (pyzbar handles both)
# ---------------------------------------------------------------------------

def detect_qr_barcodes(pil_images: list[Image.Image]) -> list[VisualElement]:
    try:
        from pyzbar.pyzbar import decode, ZBarSymbol
    except Exception as e:
        print(f"[visual] pyzbar unavailable: {e}", file=sys.stderr)
        return []

    out: list[VisualElement] = []
    for idx, img in enumerate(pil_images, start=1):
        # pyzbar accepts PIL images directly; grayscale is more reliable + faster.
        gray = img.convert("L")
        try:
            results = decode(gray)
        except Exception as e:
            print(f"[visual] pyzbar decode failed on page {idx}: {e}", file=sys.stderr)
            continue
        for r in results:
            rect = r.rect
            bbox = [
                float(rect.left),
                float(rect.top),
                float(rect.left + rect.width),
                float(rect.top + rect.height),
            ]
            payload = r.data.decode("utf-8", errors="replace") if r.data else None
            kind = "qr" if r.type == "QRCODE" else "barcode"
            out.append(VisualElement(
                type=kind, page=idx, bbox=bbox, data=payload, confidence=0.95,
            ))
    return out


# ---------------------------------------------------------------------------
# Stamps — red/purple HSV mask + circle / rectangle contour detection
# ---------------------------------------------------------------------------

def detect_stamps(pil_images: list[Image.Image]) -> list[VisualElement]:
    try:
        import cv2
    except Exception as e:
        print(f"[visual] opencv unavailable: {e}", file=sys.stderr)
        return []

    out: list[VisualElement] = []
    for idx, img in enumerate(pil_images, start=1):
        arr = np.array(img.convert("RGB"))
        # Convert to BGR for OpenCV
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Red (two hue ranges) + purple/violet — typical ink colors on Indian
        # hospital stamps.
        red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        purple = cv2.inRange(hsv, np.array([125, 50, 50]), np.array([160, 255, 255]))
        mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), purple)

        # Close small gaps so dashed stamp outlines become solid regions.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = mask.shape
        page_area = h_img * w_img

        for c in contours:
            area = cv2.contourArea(c)
            # Plausible stamp size: 0.05% — 5% of page area.
            if area < page_area * 0.0005 or area > page_area * 0.05:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h else 0
            # Stamps are roughly square/circular; reject extreme aspect ratios.
            if aspect < 0.3 or aspect > 3.5:
                continue
            # Density check: the colored pixels should fill enough of the bbox.
            bbox_pixels = w * h
            density = area / bbox_pixels if bbox_pixels else 0
            if density < 0.15:
                continue

            out.append(VisualElement(
                type="stamp",
                page=idx,
                bbox=[float(x), float(y), float(x + w), float(y + h)],
                data=None,
                confidence=min(0.9, 0.5 + density),
            ))
    return out


# ---------------------------------------------------------------------------
# Signatures — ink-density heuristic in the lower portion of each page
# ---------------------------------------------------------------------------

def detect_signatures(pil_images: list[Image.Image]) -> list[VisualElement]:
    try:
        import cv2
    except Exception:
        return []

    out: list[VisualElement] = []
    for idx, img in enumerate(pil_images, start=1):
        arr = np.array(img.convert("L"))
        h, w = arr.shape
        # Search only the bottom 30% — where signatures live on forms.
        y0 = int(h * 0.7)
        roi = arr[y0:, :]

        # Binarize: dark strokes on light background.
        _, binarized = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)

        # Morphological close to glue strokes together into stroke blobs.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Signature heuristics: wider than tall, non-trivial size, not the
            # full page width (that's a line / body text).
            if cw < 60 or ch < 15:
                continue
            if cw > w * 0.8:
                continue
            if ch > h * 0.08:
                continue
            aspect = cw / float(ch)
            if aspect < 1.5 or aspect > 12:
                continue

            # Stroke variance: real signatures have irregular strokes, not
            # uniform like printed text. Measure via contour perimeter^2 / area.
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if area < 50:
                continue
            irregularity = (perim * perim) / (area + 1)
            if irregularity < 25:  # too smooth — probably a printed underline
                continue

            bbox = [float(x), float(y0 + y), float(x + cw), float(y0 + y + ch)]
            out.append(VisualElement(
                type="signature",
                page=idx,
                bbox=bbox,
                data=None,
                confidence=min(0.8, 0.3 + irregularity / 200),
            ))
    return out


# ---------------------------------------------------------------------------
# Aggregate — called from extractor.py
# ---------------------------------------------------------------------------

def detect_all_visuals(pil_images: list[Image.Image]) -> list[VisualElement]:
    if not pil_images:
        return []
    out: list[VisualElement] = []
    try:
        out.extend(detect_qr_barcodes(pil_images))
    except Exception as e:
        print(f"[visual] QR/barcode pass failed: {e}", file=sys.stderr)
    try:
        out.extend(detect_stamps(pil_images))
    except Exception as e:
        print(f"[visual] stamp pass failed: {e}", file=sys.stderr)
    try:
        out.extend(detect_signatures(pil_images))
    except Exception as e:
        print(f"[visual] signature pass failed: {e}", file=sys.stderr)
    return out
