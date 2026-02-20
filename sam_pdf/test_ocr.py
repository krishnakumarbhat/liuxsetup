#!/usr/bin/env python3
"""Quick test to find a working PaddleOCR approach on this system."""
import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['FLAGS_pir_apply_inplace_pass'] = '0'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import paddle
paddle.set_device('cpu')

import fitz
import numpy as np
from paddleocr import PaddleOCR

# Load one page
doc = fitz.open("pdf/calculus.pdf")
page = doc[0]
zoom = 3
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat, alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

# Save as temp image for alternative test
from PIL import Image
pil_img = Image.fromarray(img)
pil_img.save("/tmp/test_page.png")
print(f"Image size: {img.shape}")

print("\n=== Test 1: PaddleOCR predict() with numpy array ===")
ocr = PaddleOCR(lang='en')
try:
    results = list(ocr.predict(img))
    print(f"predict(numpy) returned {len(results)} result(s)")
    for r in results:
        if hasattr(r, 'json'):
            rj = r.json if isinstance(r.json, dict) else {}
            res = rj.get('res', rj)
            texts = res.get('rec_texts', [])
            print(f"  Found {len(texts)} text regions")
            for t in texts[:3]:
                print(f"    -> {t}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== Test 2: PaddleOCR predict() with file path ===")
try:
    results = list(ocr.predict("/tmp/test_page.png"))
    print(f"predict(path) returned {len(results)} result(s)")
    for r in results:
        if hasattr(r, 'json'):
            rj = r.json if isinstance(r.json, dict) else {}
            res = rj.get('res', rj)
            texts = res.get('rec_texts', [])
            print(f"  Found {len(texts)} text regions")
            for t in texts[:3]:
                print(f"    -> {t}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== Test 3: PaddleOCR legacy ocr() with numpy array ===")
try:
    result = ocr.ocr(img)
    if result and result[0]:
        print(f"ocr(numpy) found {len(result[0])} text regions")
        for line in result[0][:3]:
            print(f"    -> {line[1][0]}")
    else:
        print("ocr(numpy) returned empty")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== Test 4: PaddleOCR legacy ocr() with file path ===")
try:
    result = ocr.ocr("/tmp/test_page.png")
    if result and result[0]:
        print(f"ocr(path) found {len(result[0])} text regions")
        for line in result[0][:3]:
            print(f"    -> {line[1][0]}")
    else:
        print("ocr(path) returned empty")
except Exception as e:
    print(f"FAILED: {e}")

doc.close()
print("\nDone!")
