#!/usr/bin/env python3
"""
GPU-Accelerated OCR with Invisible Text Layer
==============================================
Uses PaddleOCR (PP-OCRv4 Server) for text extraction and pyspellchecker.
Overlays invisible text on PDF for searchability.
"""

import os
import time
import argparse
import logging
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path

# Disable PaddleOCR model source check for faster startup
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR
from spellchecker import SpellChecker

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def setup_args():
    parser = argparse.ArgumentParser(description="GPU-Accelerated OCR with Invisible Text Layer")
    parser.add_argument("input_dir", type=str, help="Root directory to search for PDFs")
    parser.add_argument("output_dir", type=str, help="Directory to replicate structure and save PDFs")
    parser.add_argument("--lang", type=str, default="en", help="Language (default: en)")
    return parser.parse_args()

def fix_spelling(text, spell):
    """
    Corrects spelling but strictly ignores math/numbers to prevent breaking formulas.
    """
    if not text: 
        return text
    
    words = text.split()
    corrected = []
    for w in words:
        # Ignore short words, numbers, or words with mixed alphanumeric (e.g., 'v5', 'x2')
        if len(w) < 4 or any(c.isdigit() for c in w):
            corrected.append(w)
        # Check if word is misspelled
        elif spell.unknown([w]):
            candidate = spell.correction(w)
            corrected.append(candidate if candidate else w)
        else:
            corrected.append(w)
    return " ".join(corrected)

def process_single_pdf(pdf_path, output_path, ocr_engine, spell_engine):
    t0_file = time.time()
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        modified = False
        
        logging.info(f"Processing: {pdf_path.name} ({total_pages} pages)")

        for i, page in enumerate(doc):
            t0_page = time.time()
            
            # --- 1. Extract Image at High DPI (Zoom=3 -> ~216 DPI) ---
            zoom = 3
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Efficient conversion
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

            # --- 2. OCR with PaddleOCR ---
            # result structure: [ [box, (text, score)], ... ]
            try:
                # cls=True enables rotation correction
                result_raw = ocr_engine.ocr(img_data, cls=True)
                
                # Check if result is valid
                if result_raw and result_raw[0]:
                    result = result_raw[0]
                else:
                    result = []
                    
            except Exception as e:
                logging.warning(f"OCR error on page {i+1}: {e}")
                result = []
            
            # Explicit memory cleanup
            del img_data
            del pix

            if not result:
                continue

            # --- 3. Overlay Corrected Text ---
            for line in result:
                box = line[0]
                raw_text = line[1][0]
                confidence = line[1][1]

                # Run spell check
                final_text = fix_spelling(raw_text, spell_engine)

                # Coordinate Mapping: Map 3x scaled image coords back to PDF 1x coords
                x_min = min(p[0] for p in box) / zoom
                y_min = min(p[1] for p in box) / zoom
                y_max = max(p[1] for p in box) / zoom
                box_height = y_max - y_min

                # Insert invisible text (render_mode=3)
                text_point = fitz.Point(x_min, y_max)
                
                try:
                    page.insert_text(
                        text_point,
                        final_text,
                        fontsize=max(4, min(box_height, 20)),  # Clamp font size
                        render_mode=3,
                    )
                    modified = True
                except Exception as e:
                    pass # Ignore text insertion errors
            
            # logging.debug(f"   Page {i+1}/{total_pages} done")

        # --- 4. Save & Compress ---
        if modified:
            doc.save(output_path, garbage=4, deflate=True)
            elapsed = time.time() - t0_file
            logging.info(f"✅ DONE: {pdf_path.name} | Time: {elapsed:.2f}s | Saved")
        else:
            doc.save(output_path)
            logging.warning(f"⚠️ SKIPPED (No text): {pdf_path.name}")

        doc.close()

    except Exception as e:
        logging.error(f"❌ ERROR processing {pdf_path.name}: {e}")

def init_paddleocr(lang='en'):
    """Initialize PaddleOCR with High-Accuracy Server Models."""
    logging.info("🚀 Initializing PaddleOCR (High Accuracy Server Model)...")
    
    # PP-OCRv4 is the current state-of-the-art included in the package.
    # 'det_db_box_thresh' helps catch faint text.
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        ocr_version='PP-OCRv4',  # Force latest version
        use_gpu=True,            # Force GPU
        gpu_mem=4000,            # Allocate 4GB VRAM
        det_db_box_thresh=0.3,   # Lower threshold to catch faint math/text
        show_log=False
    )
    
    logging.info("✅ PaddleOCR initialized")
    return ocr

def main():
    args = setup_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        logging.error(f"❌ Input directory not found: {input_root}")
        return

    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Initialize Engines ---
    ocr = init_paddleocr(args.lang)
    
    logging.info("📖 Initializing Spell Checker...")
    spell = SpellChecker()

    # --- Recursive Discovery ---
    all_pdfs = list(input_root.rglob("*.pdf"))
    logging.info(f"📚 Found {len(all_pdfs)} PDFs. Starting batch processing...")

    total_start = time.time()

    for pdf in all_pdfs:
        rel_path = pdf.relative_to(input_root)
        dest_path = output_root / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        process_single_pdf(pdf, dest_path, ocr, spell)

    total_time = time.time() - total_start
    logging.info(f"🏁 All tasks completed in {total_time:.2f}s")

if __name__ == "__main__":
    main()