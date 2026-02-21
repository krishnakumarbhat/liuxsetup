# PaddleOCR PDF Searchable Layer (Batch)

This project now uses **only PaddleOCR** to process PDFs recursively and create output PDFs with an **invisible text layer**.

## Main script

- `paddle_ocr_pdf_batch.py`

## Default behavior

- Input root: `/media/pope/projecteo/github_proj/liuxsetup/tools/sam_pdf/pdf`
- Output root: `/media/pope/projecteo/github_proj/liuxsetup/tools/sam_pdf/out`
- Recursively finds all `.pdf` files under input root
- Preserves folder structure in output root
- Adds invisible searchable text to each page

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU, install a CUDA-compatible PaddlePaddle build in your environment if needed.

## Run

Use defaults exactly matching your requested path:

```bash
python3 paddle_ocr_pdf_batch.py
```

Custom paths:

```bash
python3 paddle_ocr_pdf_batch.py --input-dir /path/to/pdf --output-dir /path/to/out
```

Useful options:

```bash
python3 paddle_ocr_pdf_batch.py --lang en --dpi 400 --min-score 0.62 --device auto
```

- `--device auto` chooses GPU if available, else CPU
- `--device gpu` forces GPU
- `--device cpu` forces CPU
- `--min-score` filters low-confidence OCR to improve effective accuracy

## Utility scripts

- `paddle_cuda_probe.py`: show Paddle CUDA visibility
- `paddle_ocr_gpu_check.py`: run one-page OCR GPU smoke test
- `paddle_ocr_api_check.py`: compare `predict()` and legacy `ocr()` API output
