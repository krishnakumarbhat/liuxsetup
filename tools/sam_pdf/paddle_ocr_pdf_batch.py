#!/usr/bin/env python3
"""
Batch OCR for PDFs using PaddleOCR, preserving layout and adding an invisible
searchable text layer to output PDFs.

Default input path:
    /media/pope/projecteo/github_proj/liuxsetup/tools/sam_pdf/pdf

Default output path:
    /media/pope/projecteo/github_proj/liuxsetup/tools/sam_pdf/out
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import fitz
import numpy as np
from paddleocr import PaddleOCR

DEFAULT_INPUT_DIR = Path("/media/pope/projecteo/github_proj/liuxsetup/tools/sam_pdf/pdf")
DEFAULT_OUTPUT_DIR = Path("/media/pope/projecteo/github_proj/liuxsetup/tools/sam_pdf/out")

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("paddle_ocr_pdf_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively OCR PDFs and write searchable copies with invisible text.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input root containing PDFs recursively (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root for processed PDFs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--lang", default="en", help="PaddleOCR language code (default: en)")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for OCR (default: 300)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.55,
        help="Minimum OCR confidence to embed text (default: 0.55)",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    return parser.parse_args()


def detect_gpu_available() -> bool:
    try:
        import paddle

        if not paddle.device.is_compiled_with_cuda():
            return False

        if paddle.device.cuda.device_count() <= 0:
            return False

        cudnn_candidates = [
            "libcudnn.so",
            "libcudnn.so.9",
            "libcudnn.so.8",
            "libcudnn_cnn_infer.so.8",
            "libcudnn_ops_infer.so.8",
        ]
        for lib_name in cudnn_candidates:
            try:
                ctypes.CDLL(lib_name)
                return True
            except OSError:
                continue

        logger.warning("CUDA detected but cuDNN runtime library is not loadable. Using CPU.")
        return False
    except Exception:
        return False


def build_ocr(lang: str, device_mode: str) -> PaddleOCR:
    if device_mode == "auto":
        device = "gpu:0" if detect_gpu_available() else "cpu"
    elif device_mode == "gpu":
        device = "gpu:0"
    else:
        device = "cpu"

    logger.info("Initializing PaddleOCR (lang=%s, device=%s)", lang, device)

    def _create_ocr(target_device: str) -> PaddleOCR:
        try:
            return PaddleOCR(lang=lang, device=target_device, use_textline_orientation=True)
        except TypeError:
            try:
                return PaddleOCR(lang=lang, device=target_device)
            except TypeError:
                return PaddleOCR(lang=lang)

    return _create_ocr(device)


def find_pdfs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")


def normalize_predict_payload(payload: object) -> list[tuple[list[list[float]], str, float]]:
    if not isinstance(payload, dict):
        return []

    res = payload.get("res", payload)
    polygons = res.get("dt_polys") if isinstance(res, dict) else None
    texts = res.get("rec_texts") if isinstance(res, dict) else None
    scores = res.get("rec_scores") if isinstance(res, dict) else None

    if not isinstance(polygons, list) or not isinstance(texts, list):
        return []

    regions: list[tuple[list[list[float]], str, float]] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip() or index >= len(polygons):
            continue
        poly = polygons[index]
        score = 1.0
        if isinstance(scores, list) and index < len(scores):
            try:
                score = float(scores[index])
            except Exception:
                score = 1.0
        regions.append((poly, text.strip(), score))

    return regions


def normalize_ocr_results(ocr_engine: PaddleOCR, image: np.ndarray) -> list[tuple[list[list[float]], str, float]]:
    regions: list[tuple[list[list[float]], str, float]] = []

    try:
        if hasattr(ocr_engine, "predict"):
            for item in ocr_engine.predict(image):
                if hasattr(item, "json"):
                    regions.extend(normalize_predict_payload(item.json))
                elif isinstance(item, dict):
                    regions.extend(normalize_predict_payload(item))
    except Exception as exc:
        logger.debug("predict() failed, fallback to ocr(): %s", exc)

    if regions:
        return regions

    try:
        raw = ocr_engine.ocr(image)
        if raw and raw[0]:
            for entry in raw[0]:
                if not isinstance(entry, list) or len(entry) < 2:
                    continue
                poly = entry[0]
                text_and_score = entry[1]
                if not isinstance(text_and_score, (list, tuple)) or len(text_and_score) < 1:
                    continue
                text = str(text_and_score[0]).strip()
                if not text:
                    continue
                score = float(text_and_score[1]) if len(text_and_score) > 1 else 1.0
                regions.append((poly, text, score))
    except Exception as exc:
        logger.warning("Both OCR APIs failed: %s", exc)

    return regions


def polygon_to_rect(poly: Iterable[Iterable[float]], zoom: float) -> fitz.Rect | None:
    points = list(poly)
    if not points:
        return None

    xs: list[float] = []
    ys: list[float] = []

    for point in points:
        coords = list(point)
        if len(coords) < 2:
            continue
        xs.append(float(coords[0]) / zoom)
        ys.append(float(coords[1]) / zoom)

    if not xs or not ys:
        return None

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    if x1 - x0 < 1 or y1 - y0 < 1:
        return None

    return fitz.Rect(x0, y0, x1, y1)


def add_invisible_text(page: fitz.Page, rect: fitz.Rect, text: str) -> bool:
    if not text.strip():
        return False

    font_size = max(4.0, min(rect.height * 0.9, 28.0))
    inserted = page.insert_textbox(
        rect,
        text,
        fontsize=font_size,
        render_mode=3,
        overlay=True,
    )

    if inserted < 0:
        baseline = fitz.Point(rect.x0, rect.y1)
        page.insert_text(baseline, text, fontsize=font_size, render_mode=3, overlay=True)

    return True


def process_pdf(
    pdf_path: Path,
    out_path: Path,
    ocr_engine: PaddleOCR,
    dpi: int,
    min_score: float,
) -> None:
    start = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open %s: %s", pdf_path, exc)
        return

    zoom = dpi / 72.0
    modified = False
    total_regions = 0

    for page_index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

        regions = normalize_ocr_results(ocr_engine, image)

        page_inserted = 0
        for poly, text, score in regions:
            if score < min_score:
                continue
            rect = polygon_to_rect(poly, zoom)
            if rect is None:
                continue
            if add_invisible_text(page, rect, text):
                modified = True
                page_inserted += 1

        total_regions += page_inserted
        logger.info("%s page %d: inserted %d text regions", pdf_path.name, page_index, page_inserted)

    if modified:
        doc.save(out_path, garbage=4, deflate=True)
    else:
        doc.close()
        shutil.copy2(pdf_path, out_path)
        logger.warning("No text detected in %s; copied original", pdf_path)
        return

    doc.close()
    logger.info(
        "Done %s -> %s | regions=%d | %.2fs",
        pdf_path,
        out_path,
        total_regions,
        time.time() - start,
    )


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_engine = build_ocr(lang=args.lang, device_mode=args.device)
    pdfs = find_pdfs(input_dir)
    logger.info("Found %d PDF file(s) in %s", len(pdfs), input_dir)

    for pdf_path in pdfs:
        relative = pdf_path.relative_to(input_dir)
        out_path = output_dir / relative
        process_pdf(pdf_path, out_path, ocr_engine, dpi=args.dpi, min_score=args.min_score)

    logger.info("All done. Output root: %s", output_dir)


if __name__ == "__main__":
    main()
