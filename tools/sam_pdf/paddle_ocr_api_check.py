#!/usr/bin/env python3
"""Check PaddleOCR API behavior (predict vs ocr) on a single PDF page."""

from __future__ import annotations

import argparse
import os

import fitz
import numpy as np
from paddleocr import PaddleOCR

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR API check")
    parser.add_argument("--pdf", default="pdf/calculus.pdf", help="Input PDF path")
    parser.add_argument("--page", type=int, default=1, help="1-based page number")
    parser.add_argument("--lang", default="en", help="PaddleOCR language code")
    return parser.parse_args()


def load_page_image(pdf_path: str, page_number: int) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page_index = max(0, min(page_number - 1, len(doc) - 1))
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return image


def main() -> None:
    args = parse_args()
    image = load_page_image(args.pdf, args.page)
    ocr = PaddleOCR(lang=args.lang)

    print("=== predict() ===")
    try:
        count = 0
        if hasattr(ocr, "predict"):
            for item in ocr.predict(image):
                if hasattr(item, "json") and isinstance(item.json, dict):
                    res = item.json.get("res", item.json)
                    texts = res.get("rec_texts", [])
                    for text in texts[:5]:
                        print("-", text)
                    count += len(texts)
        print(f"predict() regions: {count}")
    except Exception as exc:
        print(f"predict() failed: {exc}")

    print("\n=== ocr() ===")
    try:
        legacy = ocr.ocr(image)
        if legacy and legacy[0]:
            for line in legacy[0][:5]:
                print("-", line[1][0])
            print(f"ocr() regions: {len(legacy[0])}")
        else:
            print("ocr() regions: 0")
    except Exception as exc:
        print(f"ocr() failed: {exc}")


if __name__ == "__main__":
    main()
