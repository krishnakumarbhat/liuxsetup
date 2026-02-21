#!/usr/bin/env python3
"""GPU smoke test for PaddleOCR on the first page of a PDF."""

from __future__ import annotations

import argparse
import os

import fitz
import numpy as np
import paddle
from paddleocr import PaddleOCR

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-page PaddleOCR GPU test")
    parser.add_argument("--pdf", default="pdf/calculus.pdf", help="Input PDF path")
    parser.add_argument("--page", type=int, default=1, help="1-based page number")
    parser.add_argument("--lang", default="en", help="PaddleOCR language code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    compiled_with_cuda = paddle.device.is_compiled_with_cuda()
    cuda_devices = paddle.device.cuda.device_count() if compiled_with_cuda else 0
    use_gpu = compiled_with_cuda and cuda_devices > 0

    print(f"Paddle version: {paddle.__version__}")
    print(f"CUDA compiled: {compiled_with_cuda}")
    print(f"CUDA devices: {cuda_devices}")

    ocr = PaddleOCR(lang=args.lang, use_gpu=use_gpu)

    doc = fitz.open(args.pdf)
    page_index = max(0, min(args.page - 1, len(doc) - 1))
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    regions = []
    if hasattr(ocr, "predict"):
        for item in ocr.predict(image):
            if hasattr(item, "json") and isinstance(item.json, dict):
                res = item.json.get("res", item.json)
                texts = res.get("rec_texts", [])
                regions.extend(texts)
    if not regions:
        legacy = ocr.ocr(image)
        if legacy and legacy[0]:
            regions = [line[1][0] for line in legacy[0]]

    print(f"Detected text regions: {len(regions)}")
    for text in regions[:10]:
        print("-", text)

    doc.close()


if __name__ == "__main__":
    main()
