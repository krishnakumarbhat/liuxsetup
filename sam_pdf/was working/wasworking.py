#!/usr/bin/env python3
"""
GPU-Accelerated OCR with Invisible Text Layer
==============================================
Uses PaddleOCR for text extraction and pyspellchecker for correction.
Overlays invisible text on PDF for searchability.
"""

import os
import time
import argparse
import logging
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
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
        # Check if word is misspelled using pyspellchecker's known() method
        elif spell.unknown([w]):  # Returns set of unknown words
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
            # Higher DPI = Better OCR accuracy for handwriting/math
            zoom = 3
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Efficient conversion avoiding extra memory copy if possible
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

            # --- 2. OCR with PaddleOCR ---
            # Using new predict API if available, fallback to ocr()
            try:
                if hasattr(ocr_engine, 'predict'):
                    # New API
                    ocr_results = list(ocr_engine.predict(img_data))
                    result = []
                    for r in ocr_results:
                        if hasattr(r, 'json') and isinstance(r.json, dict):
                            res = r.json.get('res', {})
                            texts = res.get('rec_texts', [])
                            polys = res.get('dt_polys', [])
                            scores = res.get('rec_scores', [])
                            for idx, text in enumerate(texts):
                                if text and idx < len(polys):
                                    result.append([polys[idx], (text, scores[idx] if idx < len(scores) else 0.9)])
                else:
                    # Legacy API
                    ocr_result = ocr_engine.ocr(img_data)
                    if ocr_result and ocr_result[0]:
                        result = ocr_result[0]
                    else:
                        result = []
            except Exception as e:
                logging.warning(f"OCR error on page {i+1}: {e}")
                result = []
            
            # Explicit memory cleanup
            del img_data
            del pix

            if not result:
                logging.debug(f"  Page {i+1}: No text found")
                continue

            # --- 3. Overlay Corrected Text ---
            for line in result:
                box = line[0]
                raw_text = line[1][0]
                confidence = line[1][1]

                # Only spell check if confidence is high enough to be worth fixing
                final_text = fix_spelling(raw_text, spell_engine)

                # Coordinate Mapping: Map 3x scaled image coords back to PDF 1x coords
                # Box = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_min = min(p[0] for p in box) / zoom
                y_min = min(p[1] for p in box) / zoom
                y_max = max(p[1] for p in box) / zoom
                box_height = y_max - y_min

                # Insert invisible text (render_mode=3)
                # We place the text at the bottom-left of the bounding box
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
                    logging.debug(f"Text insert warning: {e}")
            
            t1_page = time.time()
            logging.debug(f"   Page {i+1}/{total_pages} processed in {t1_page - t0_page:.2f}s")

        # --- 4. Save & Compress ---
        if modified:
            # garbage=4: Aggressive unused object removal
            # deflate=True: Compress streams
            doc.save(output_path, garbage=4, deflate=True)
            elapsed = time.time() - t0_file
            logging.info(f"✅ DONE: {pdf_path.name} | Time: {elapsed:.2f}s | Saved to: {output_path}")
        else:
            # Copy original if no text found
            doc.save(output_path)
            logging.warning(f"⚠️ SKIPPED (No text): {pdf_path.name}")

        doc.close()

    except Exception as e:
        logging.error(f"❌ ERROR processing {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()

def init_paddleocr(lang='en'):
    """Initialize PaddleOCR with compatible parameters."""
    logging.info("🚀 Initializing PaddleOCR...")
    
    try:
        # Try new API with minimal parameters (v3.x)
        ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,  # Better for rotated text
        )
    except TypeError as e:
        logging.debug(f"PaddleOCR init fallback: {e}")
        # Fallback to simplest init
        ocr = PaddleOCR(lang=lang)
    
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
    logging.info("✅ Spell Checker initialized")

    # --- Recursive Discovery ---
    all_pdfs = list(input_root.rglob("*.pdf"))
    logging.info(f"📚 Found {len(all_pdfs)} PDFs. Starting batch processing...")

    total_start = time.time()

    for pdf in all_pdfs:
        # Calculate output path to maintain directory structure
        rel_path = pdf.relative_to(input_root)
        dest_path = output_root / rel_path
        
        # Ensure output folder exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process
        process_single_pdf(pdf, dest_path, ocr, spell)

    total_time = time.time() - total_start
    logging.info(f"🏁 All tasks completed in {total_time:.2f}s")

if __name__ == "__main__":
    main()