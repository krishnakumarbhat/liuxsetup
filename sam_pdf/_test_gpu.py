#!/usr/bin/env python3
"""Quick GPU OCR test on one page of calculus.pdf"""
import os, ctypes
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Load cuDNN before paddle imports
cudnn_dir = "/media/pope/projecteo/python_libs/site-packages/nvidia/cudnn/lib"
cublas_dir = "/media/pope/projecteo/python_libs/site-packages/nvidia/cublas/lib"
os.environ['LD_LIBRARY_PATH'] = f"{cudnn_dir}:{cublas_dir}:" + os.environ.get('LD_LIBRARY_PATH', '')
# Pre-load cuDNN so paddle can find it
for lib in ['libcudnn_ops_infer.so.8', 'libcudnn_cnn_infer.so.8', 'libcudnn.so.8']:
    try:
        ctypes.CDLL(os.path.join(cudnn_dir, lib), mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Warning loading {lib}: {e}")

import paddle
paddle.set_device('gpu:0')
print(f"Paddle {paddle.__version__}, GPU: {paddle.device.is_compiled_with_cuda()}")

from paddleocr import PaddleOCR
print("Importing PaddleOCR done")

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
print("PaddleOCR initialized")

import fitz
import numpy as np
import time

doc = fitz.open('pdf/calculus.pdf')
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
print(f"Image shape: {img.shape}")

print("Running OCR...")
t0 = time.time()
result = ocr.ocr(img)
t1 = time.time()
print(f"OCR took {t1-t0:.2f}s")

if result and result[0]:
    print(f"\nFOUND {len(result[0])} text regions on page 1:")
    for i, line in enumerate(result[0][:8]):
        print(f"  [{i}] text='{line[1][0]}'  conf={line[1][1]:.3f}")
else:
    print(f"NO TEXT FOUND. result={result}")

doc.close()
print("\nDONE")
