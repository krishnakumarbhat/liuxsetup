#!/usr/bin/env python3
"""
Maximum-Accuracy OCR → Searchable PDF
======================================
Extracts text from scanned/image PDFs using EasyOCR with aggressive
preprocessing for maximum accuracy. Overlays invisible text layer
so the PDF becomes searchable (Ctrl+F works).

Hardware targets: i5-12th Gen, 24GB RAM, RTX 3050 (4GB VRAM)

Usage:
    python3 main2.py <input_dir> <output_dir>
    python3 main2.py <input_dir> <output_dir> --lang en --dpi 400 --workers 4

Example:
    python3 main2.py ./pdf/ ./output/
    python3 main2.py /media/pope/projecteo/OfficeLens/new/ /media/pope/projecteo/github_proj/sam_tool/out
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import os
import sys
import time
import argparse
import logging
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fitz          # PyMuPDF – PDF reading/writing
import numpy as np
import cv2           # OpenCV – image preprocessing
from PIL import Image, ImageEnhance, ImageFilter

import easyocr
from spellchecker import SpellChecker

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ocr")


# ══════════════════════════════════════════════════════════════════════
#  1.  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Max-accuracy OCR: scanned PDF → searchable PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_dir",  help="Folder with source PDFs (searched recursively)")
    p.add_argument("output_dir", help="Folder to write searchable PDFs (structure preserved)")
    p.add_argument("--lang",    default="en",  help="OCR language (default: en)")
    p.add_argument("--dpi",     type=int, default=400, help="Render DPI – higher = more accurate but slower (default: 400)")
    p.add_argument("--workers", type=int, default=4,   help="Threads for image preprocessing (default: 4)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  2.  IMAGE PREPROCESSING  (maximize OCR accuracy)
# ══════════════════════════════════════════════════════════════════════
def preprocess_for_ocr(img_np: np.ndarray) -> np.ndarray:
    """
    Apply a pipeline of image enhancements to maximise OCR accuracy.
    Input:  RGB uint8 numpy array
    Output: RGB uint8 numpy array (enhanced)

    Pipeline:
      1. Convert to grayscale → denoise → sharpen → adaptive threshold
      2. Convert back to RGB (EasyOCR expects 3-channel)
    """
    # --- Step 1: Grayscale ---
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # --- Step 2: Denoise (Non-local means – slow but preserves edges) ---
    # h=10 : filter strength.  Larger h removes more noise but removes detail too.
    # templateWindowSize=7, searchWindowSize=21
    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)

    # --- Step 3: CLAHE – adaptive contrast enhancement ---
    # Improves readability of faded handwriting / low-contrast scans
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    # --- Step 4: Sharpen with unsharp mask ---
    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(contrast, 1.5, blurred, -0.5, 0)

    # --- Step 5: Adaptive threshold (binarise) for cleaner text ---
    # Use a large block size for handwritten / uneven lighting
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    # --- Step 6: Morphological close – fill tiny gaps in strokes ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # --- Convert back to 3-channel RGB (EasyOCR requirement) ---
    rgb_out = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    return rgb_out


def preprocess_light(img_np: np.ndarray) -> np.ndarray:
    """
    Lighter preprocessing – just contrast + sharpen, no binarisation.
    Used as a second pass to catch text that binarisation might lose.
    """
    pil = Image.fromarray(img_np)
    pil = ImageEnhance.Contrast(pil).enhance(1.6)
    pil = ImageEnhance.Sharpness(pil).enhance(2.0)
    return np.array(pil)


# ══════════════════════════════════════════════════════════════════════
#  3.  OCR ENGINE INITIALISATION
# ══════════════════════════════════════════════════════════════════════
def init_ocr(lang: str = "en") -> easyocr.Reader:
    """Create an EasyOCR Reader, using GPU if available."""
    log.info("🚀 Initialising EasyOCR …")

    lang_map = {
        "en": ["en"], "ch": ["ch_sim", "en"], "ja": ["ja", "en"],
        "ko": ["ko", "en"], "fr": ["fr", "en"], "de": ["de", "en"],
        "es": ["es", "en"], "ar": ["ar", "en"], "hi": ["hi", "en"],
    }
    lang_list = lang_map.get(lang, [lang])

    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_mem / 1024**3
            log.info(f"   GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    except Exception:
        pass

    reader = easyocr.Reader(
        lang_list,
        gpu=use_gpu,
        model_storage_directory=None,   # default cache
        download_enabled=True,
    )
    log.info(f"✅ EasyOCR ready  │ langs={lang_list}  GPU={use_gpu}")
    return reader


# ══════════════════════════════════════════════════════════════════════
#  4.  SPELL CHECKER
# ══════════════════════════════════════════════════════════════════════
def init_spell() -> SpellChecker:
    log.info("📖 Initialising spell checker …")
    sp = SpellChecker()
    log.info("✅ Spell checker ready")
    return sp


def fix_spelling(text: str, spell: SpellChecker) -> str:
    """
    Correct misspellings but NEVER touch:
      • words shorter than 4 chars  (likely abbreviations / math)
      • words containing digits      (formulas, codes, IDs)
      • words that are ALL CAPS       (acronyms like HTTP, API)
      • words with special chars      (file paths, URLs)
    """
    if not text:
        return text
    tokens = text.split()
    out = []
    for w in tokens:
        if (len(w) < 4
            or any(c.isdigit() for c in w)
            or w.isupper()
            or not w.isalpha()):
            out.append(w)
        elif spell.unknown([w.lower()]):
            fix = spell.correction(w.lower())
            out.append(fix if fix else w)
        else:
            out.append(w)
    return " ".join(out)


# ══════════════════════════════════════════════════════════════════════
#  5.  MULTI-PASS OCR ON A SINGLE PAGE
# ══════════════════════════════════════════════════════════════════════
def ocr_page_multipass(reader: easyocr.Reader, img_np: np.ndarray):
    """
    Run OCR with TWO preprocessing passes and merge results.
    This catches text that one preprocessing mode might miss.

    Returns list of (bbox, text, confidence) – deduplicated.
    """
    # --- Pass 1: Heavy preprocessing (binarised) ---
    img_heavy = preprocess_for_ocr(img_np)
    results_heavy = reader.readtext(
        img_heavy,
        detail=1,
        paragraph=False,
        min_size=10,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.3,
        width_ths=0.7,
        decoder="beamsearch",     # slower but more accurate than greedy
        beamWidth=10,
    )

    # --- Pass 2: Light preprocessing (contrast + sharpen only) ---
    img_light = preprocess_light(img_np)
    results_light = reader.readtext(
        img_light,
        detail=1,
        paragraph=False,
        min_size=10,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.3,
        width_ths=0.7,
        decoder="beamsearch",
        beamWidth=10,
    )

    # --- Merge: keep the higher-confidence result for overlapping boxes ---
    merged = {}  # key = rounded (x_center, y_center)
    for results in [results_heavy, results_light]:
        for (box, text, conf) in results:
            if not text or not text.strip():
                continue
            cx = int(np.mean([p[0] for p in box]) / 20) * 20  # quantise to 20px
            cy = int(np.mean([p[1] for p in box]) / 20) * 20
            key = (cx, cy)
            if key not in merged or conf > merged[key][2]:
                merged[key] = (box, text, conf)

    return list(merged.values())


# ══════════════════════════════════════════════════════════════════════
#  6.  PROCESS ONE PDF
# ══════════════════════════════════════════════════════════════════════
def process_pdf(
    pdf_path: Path,
    output_path: Path,
    reader: easyocr.Reader,
    spell: SpellChecker,
    dpi: int = 400,
):
    """
    Open a PDF, OCR every page at high DPI with multi-pass preprocessing,
    overlay invisible text, and save the searchable result.
    """
    t0 = time.time()
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log.error(f"❌ Cannot open {pdf_path.name}: {e}")
        return

    n_pages = len(doc)
    log.info(f"📄 {pdf_path.name}  │  {n_pages} page(s)  │  DPI={dpi}")
    modified = False

    for page_idx in range(n_pages):
        page = doc[page_idx]
        tp = time.time()

        # ── Render page to image at high DPI ──
        zoom = dpi / 72.0      # fitz default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )

        # ── Multi-pass OCR ──
        try:
            results = ocr_page_multipass(reader, img)
        except Exception as e:
            log.warning(f"   ⚠ OCR error p{page_idx+1}: {e}")
            results = []

        # Free heavy memory immediately
        del img, pix
        gc.collect()

        if not results:
            log.debug(f"   p{page_idx+1}: no text")
            continue

        # ── Overlay invisible text ──
        n_inserted = 0
        for (box, raw_text, confidence) in results:
            text = fix_spelling(raw_text, spell)

            # Map image coords → PDF coords  (divide by zoom)
            x_min = min(p[0] for p in box) / zoom
            y_min = min(p[1] for p in box) / zoom
            y_max = max(p[1] for p in box) / zoom
            box_h = y_max - y_min

            try:
                page.insert_text(
                    fitz.Point(x_min, y_max),
                    text,
                    fontsize=max(4, min(box_h, 20)),
                    render_mode=3,       # invisible
                )
                n_inserted += 1
                modified = True
            except Exception:
                pass

        elapsed_p = time.time() - tp
        log.info(
            f"   p{page_idx+1}/{n_pages}  │  {n_inserted} text regions"
            f"  │  {elapsed_p:.1f}s"
        )

    # ── Save ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if modified:
        doc.save(str(output_path), garbage=4, deflate=True)
        log.info(
            f"✅ {pdf_path.name}  │  {time.time()-t0:.1f}s"
            f"  │  → {output_path}"
        )
    else:
        doc.save(str(output_path))
        log.warning(f"⚠️  {pdf_path.name}: no text found (copied as-is)")

    doc.close()


# ══════════════════════════════════════════════════════════════════════
#  7.  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    src  = Path(args.input_dir)
    dst  = Path(args.output_dir)

    if not src.exists():
        log.error(f"❌ Input not found: {src}")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    # ── Init engines ──
    reader = init_ocr(args.lang)
    spell  = init_spell()

    # ── Discover PDFs ──
    pdfs = sorted(src.rglob("*.pdf"))
    if not pdfs:
        log.warning("No PDFs found.")
        return

    log.info(f"📚 Found {len(pdfs)} PDF(s).  DPI={args.dpi}  Workers={args.workers}")
    log.info("=" * 60)

    t_all = time.time()
    for i, pdf in enumerate(pdfs, 1):
        rel = pdf.relative_to(src)
        out = dst / rel
        log.info(f"[{i}/{len(pdfs)}] ─────────────────────────────────")
        process_pdf(pdf, out, reader, spell, dpi=args.dpi)

    log.info("=" * 60)
    log.info(f"🏁 Done!  {len(pdfs)} PDFs in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()