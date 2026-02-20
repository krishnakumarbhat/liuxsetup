#!/usr/bin/env python3
"""
PDF OCR + VLM Correction Pipeline (Optimized)
==============================================
1. Extract page image at high DPI
2. Chunk-by-chunk PaddleOCR (GPU accelerated)
3. Complete page OCR + image sent to Qwen3-VL for correction
4. Show diff between PaddleOCR and Qwen output
5. Overlay corrected text precisely on handwriting in PDF
6. Compress with deflation/garbage collection

Optimizations:
- Full NVIDIA GPU utilization (PaddleOCR + VLM)
- Efficient memory management
- Parallel chunk processing where possible
- Detailed timing metrics

Requirements:
- pip install paddleocr PyMuPDF requests Pillow difflib colorama
- Local VLM server running at http://localhost:8000
"""


1. Extract page image at high DPI
2. Chunk-by-chunk PaddleOCR (GPU accelerated)
5. Overlay corrected text precisely on handwriting in PDF
6. Compress with deflation/garbage collection

Optimizations:
- Full NVIDIA GPU utilization (PaddleOCR)
- Efficient memory management
- Parallel chunk processing where possible
- Detailed timing metrics


import os
import sys
import base64
import requests
import logging
import time
import difflib
from pathlib import Path
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Try to import colorama for colored diff output
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
INPUT_DIR = Path("/media/pope/projecteo/github_proj/sam_pdf/pdf")
OUTPUT_DIR = Path("/media/pope/projecteo/github_proj/sam_pdf/output")
DIFF_LOG_DIR = Path("/media/pope/projecteo/github_proj/sam_pdf/diff_logs")

# Local VLM Server (Transformers HTTP server)
API_BASE_URL = os.environ.get("VLM_API_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")

# Processing settings
DPI = 200  # High DPI for better OCR accuracy on handwritten text
CHUNK_HEIGHT = 400  # Pixels - chunk the page vertically for better OCR

# GPU Settings - Force CUDA for both PaddleOCR and VLM
USE_GPU = True
PADDLE_GPU_MEM = 2000  # MB - PaddleOCR GPU memory limit (leave room for VLM)

# Runtime flags
class Config:
    enable_ai = True
    vlm_error_logged = False

# Logging setup with timing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DIFF VISUALIZATION
# ============================================================================

def show_diff(original: str, corrected: str, page_num: int = 0) -> str:
    """
    Show colored diff between PaddleOCR output and Qwen corrected output.
    Returns the diff as a string (also prints to console).
    """
    original_lines = original.strip().split('\n')
    corrected_lines = corrected.strip().split('\n')
    
    diff = list(difflib.unified_diff(
        original_lines,
        corrected_lines,
        fromfile='PaddleOCR',
        tofile='Qwen-VLM',
        lineterm=''
    ))
    
    if not diff:
        logger.info(f"  📝 Page {page_num}: No differences (OCR was accurate)")
        return ""
    
    diff_output = []
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 DIFF - Page {page_num}: PaddleOCR vs Qwen-VLM")
    logger.info(f"{'='*60}")
    
    for line in diff:
        if line.startswith('+++') or line.startswith('---'):
            if HAS_COLORAMA:
                print(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
            else:
                print(f"[HEADER] {line}")
            diff_output.append(f"[HEADER] {line}")
        elif line.startswith('+'):
            if HAS_COLORAMA:
                print(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
            else:
                print(f"[+ADD] {line}")
            diff_output.append(f"[+ADD] {line}")
        elif line.startswith('-'):
            if HAS_COLORAMA:
                print(f"{Fore.RED}{line}{Style.RESET_ALL}")
            else:
                print(f"[-DEL] {line}")
            diff_output.append(f"[-DEL] {line}")
        elif line.startswith('@@'):
            if HAS_COLORAMA:
                print(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
            else:
                print(f"[LOC] {line}")
            diff_output.append(f"[LOC] {line}")
        else:
            print(f"      {line}")
            diff_output.append(f"      {line}")
    
    logger.info(f"{'='*60}\n")
    
    return '\n'.join(diff_output)

def save_diff_log(pdf_name: str, page_num: int, ocr_text: str, vlm_text: str, diff_str: str):
    """Save diff log to file for later review."""
    DIFF_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = DIFF_LOG_DIR / f"{pdf_name}_page{page_num}_{timestamp}.diff"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"PDF: {pdf_name}\n")
        f.write(f"Page: {page_num}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\n{'='*60}\n")
        f.write("PADDLEOCR OUTPUT:\n")
        f.write(f"{'='*60}\n")
        f.write(ocr_text)
        f.write(f"\n\n{'='*60}\n")
        f.write("QWEN-VLM CORRECTED OUTPUT:\n")
        f.write(f"{'='*60}\n")
        f.write(vlm_text)
        f.write(f"\n\n{'='*60}\n")
        f.write("DIFF:\n")
        f.write(f"{'='*60}\n")
        f.write(diff_str)
    
    logger.info(f"  📄 Diff saved to: {log_file}")

# ============================================================================
# PADDLEOCR INITIALIZATION (GPU Optimized)
# ============================================================================

def init_paddleocr():
    """Initialize PaddleOCR - GPU is auto-detected by PaddlePaddle."""
    os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    
    from paddleocr import PaddleOCR
    
    logger.info("🚀 Initializing PaddleOCR...")
    
    try:
        # New PaddleOCR API (v3.x)
        ocr = PaddleOCR(
            lang='en',
            use_textline_orientation=True,  # Better for rotated text
        )
    except TypeError as e:
        # Fallback for simpler init
        logger.debug(f"PaddleOCR init fallback: {e}")
        ocr = PaddleOCR(lang='en')
    
    logger.info("✅ PaddleOCR initialized")
    return ocr

# ============================================================================
# VLM API CLIENT (Optimized for Math)
# ============================================================================

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 data URI."""
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def call_vlm_server(image_bytes: bytes, ocr_text: str, timeout: int = 180) -> str:
    """
    Call local VLM server to verify/correct OCR text.
    Optimized prompt for mathematical equations and handwriting.
    """
    if not Config.enable_ai:
        return ocr_text
    
    start_time = time.time()
    
    try:
        image_uri = encode_image_to_base64(image_bytes)
        
        # Optimized system prompt for math equations and handwriting
        system_prompt = """You are a precise OCR correction assistant specialized in:
1. Mathematical equations (fractions, integrals, derivatives, summations, Greek letters)
2. Handwritten text recognition
3. Scientific notation and formulas

RULES:
- Output ONLY the corrected text, nothing else
- Preserve the EXACT line structure and spacing from input
- Fix mathematical symbols: 
  * "∫" for integrals, "∑" for summations, "∂" for partial derivatives
  * Fractions like "1/2", exponents like "x^2", subscripts like "x_i"
- Fix Greek letters: α, β, γ, δ, ε, θ, λ, μ, π, σ, ω, etc.
- Fix common OCR errors in handwriting
- Do NOT add explanations or commentary"""

        user_prompt = f"""Correct this OCR text from a scanned PDF page. The image shows the actual page content.

OCR TEXT TO CORRECT:
{ocr_text}

OUTPUT (corrected text only, same structure):"""

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ],
            "max_tokens": 1024,  # Larger for full page
            "temperature": 0.0,  # Deterministic
            "stream": False
        }
        
        logger.info(f"  📤 Sending to VLM ({len(ocr_text)} chars, ~{len(image_bytes)//1024}KB image)...")
        
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            json=payload,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            corrected = result["choices"][0]["message"]["content"].strip()
            
            # Clean up markdown code blocks
            if corrected.startswith("```") and corrected.endswith("```"):
                lines = corrected.split('\n')
                corrected = '\n'.join(lines[1:-1]).strip()
            
            # Handle thinking model verbose output
            if len(corrected) > len(ocr_text) * 5:
                # Extract actual correction from verbose output
                lines = corrected.split('\n')
                filtered = []
                skip_phrases = ["let me", "i need", "looking at", "appears to be", 
                               "corrected text:", "here is", "okay", "first", "i'll",
                               "the text shows", "i can see", "analyzing"]
                
                for line in reversed(lines):
                    line = line.strip()
                    if line and not any(p in line.lower() for p in skip_phrases):
                        filtered.insert(0, line)
                        if len('\n'.join(filtered)) >= len(ocr_text) * 0.8:
                            break
                
                if filtered:
                    corrected = '\n'.join(filtered)
                else:
                    logger.warning("  ⚠️ VLM output too verbose, using original")
                    return ocr_text
            
            # Remove common prefixes
            for prefix in ["Corrected text:", "Output:", "Here is the corrected text:",
                          "The corrected text is:", "Fixed text:", "Result:"]:
                if corrected.lower().startswith(prefix.lower()):
                    corrected = corrected[len(prefix):].strip()
            
            logger.info(f"  ✅ VLM response in {elapsed:.2f}s ({len(corrected)} chars)")
            return corrected if corrected else ocr_text
        else:
            logger.warning(f"  ❌ VLM error {response.status_code}: {response.text[:200]}")
            return ocr_text
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Lost connection to VLM server! Exiting to prevent low-quality output.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        logger.warning(f"  ⚠️ VLM timeout after {timeout}s, using OCR text")
        return ocr_text
    except Exception as e:
        logger.warning(f"  ⚠️ VLM error: {e}")
        return ocr_text

# ============================================================================
# PDF PROCESSING (Optimized with Chunking)
# ============================================================================

@dataclass
class TextBlock:
    """Represents a text block with position info."""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence: float

def extract_page_image(page: fitz.Page, dpi: int = DPI) -> bytes:
    """Extract page as PNG image bytes."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def chunk_image_vertically(image_bytes: bytes, chunk_height: int = CHUNK_HEIGHT) -> List[Tuple[bytes, int]]:
    """
    Split image into vertical chunks for better OCR accuracy.
    Returns list of (chunk_bytes, y_offset) tuples.
    """
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    
    chunks = []
    y = 0
    
    while y < height:
        end_y = min(y + chunk_height, height)
        chunk = img.crop((0, y, width, end_y))
        
        buf = BytesIO()
        chunk.save(buf, format='PNG')
        chunks.append((buf.getvalue(), y))
        
        y = end_y
    
    return chunks

def run_ocr_on_image(ocr_engine, image_bytes: bytes, y_offset: int = 0) -> List[TextBlock]:
    """
    Run PaddleOCR on image bytes.
    Returns list of TextBlock objects with adjusted y coordinates.
    """
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)
    
    results = []
    
    try:
        # Try new predict API
        if hasattr(ocr_engine, 'predict'):
            ocr_result = list(ocr_engine.predict(img_array))
            for r in ocr_result:
                if hasattr(r, 'json') and isinstance(r.json, dict):
                    res = r.json.get('res', {})
                    texts = res.get('rec_texts', [])
                    polys = res.get('dt_polys', [])
                    scores = res.get('rec_scores', [])
                    
                    for i, text in enumerate(texts):
                        if text and i < len(polys):
                            poly = polys[i]
                            xs = [p[0] for p in poly]
                            ys = [p[1] + y_offset for p in poly]  # Add offset
                            results.append(TextBlock(
                                text=text,
                                bbox=(min(xs), min(ys), max(xs), max(ys)),
                                confidence=scores[i] if i < len(scores) else 0.9
                            ))
        else:
            # Legacy ocr() API
            ocr_result = ocr_engine.ocr(img_array)
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    poly = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    xs = [p[0] for p in poly]
                    ys = [p[1] + y_offset for p in poly]
                    results.append(TextBlock(
                        text=text,
                        bbox=(min(xs), min(ys), max(xs), max(ys)),
                        confidence=conf
                    ))
    except Exception as e:
        logger.error(f"  OCR error: {e}")
    
    return results

def run_chunked_ocr(ocr_engine, page_image: bytes) -> Tuple[List[TextBlock], str]:
    """
    Run OCR on page in chunks for better accuracy.
    Returns (list of TextBlocks, full page text).
    """
    start_time = time.time()
    
    # Split into chunks
    chunks = chunk_image_vertically(page_image, CHUNK_HEIGHT)
    logger.info(f"    📐 Split page into {len(chunks)} chunks")
    
    all_blocks = []
    
    # Process each chunk
    for i, (chunk_bytes, y_offset) in enumerate(chunks):
        blocks = run_ocr_on_image(ocr_engine, chunk_bytes, y_offset)
        all_blocks.extend(blocks)
        logger.debug(f"    Chunk {i+1}: {len(blocks)} blocks")
    
    # Sort by y position then x position (reading order)
    all_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    
    # Build full page text
    full_text = '\n'.join([b.text for b in all_blocks])
    
    elapsed = time.time() - start_time
    logger.info(f"    📝 OCR complete: {len(all_blocks)} blocks in {elapsed:.2f}s")
    
    return all_blocks, full_text

def create_searchable_pdf(
    input_path: Path,
    output_path: Path,
    ocr_engine,
) -> bool:
    """
    Process PDF: Chunked OCR + VLM Correction + Diff + Overlay
    """
    total_start = time.time()
    pdf_name = input_path.stem
    
    try:
        src_doc = fitz.open(str(input_path))
        out_doc = fitz.open()
        
        total_pages = len(src_doc)
        logger.info(f"📖 Processing {total_pages} pages from: {input_path.name}")
        
        for page_num in range(total_pages):
            page_start = time.time()
            src_page = src_doc[page_num]
            
            logger.info(f"\n{'─'*50}")
            logger.info(f"📄 Page {page_num + 1}/{total_pages}")
            logger.info(f"{'─'*50}")
            
            # Create new page with same dimensions
            new_page = out_doc.new_page(
                width=src_page.rect.width,
                height=src_page.rect.height
            )
            
            # Copy original page content
            new_page.show_pdf_page(new_page.rect, src_doc, page_num)
            
            # Extract page image at high DPI
            logger.info("  📸 Extracting page image...")
            page_image = extract_page_image(src_page)
            
            # Run chunked OCR
            logger.info("  🔍 Running PaddleOCR (chunked)...")
            text_blocks, ocr_full_text = run_chunked_ocr(ocr_engine, page_image)
            
            if not text_blocks:
                logger.warning("  ⚠️ No text found on page")
                continue
            
            logger.info(f"    Found {len(text_blocks)} text blocks")
            
            # Send full page to VLM for correction
            corrected_text = ocr_full_text
            if Config.enable_ai:
                logger.info("  🤖 Sending to Qwen-VLM for correction...")
                corrected_text = call_vlm_server(page_image, ocr_full_text)
                
                # Show and save diff
                diff_str = show_diff(ocr_full_text, corrected_text, page_num + 1)
                if diff_str:
                    save_diff_log(pdf_name, page_num + 1, ocr_full_text, corrected_text, diff_str)
            
            # Split corrected text back into lines
            corrected_lines = [line.strip() for line in corrected_text.split('\n') if line.strip()]
            
            # Scale factor: OCR was done at DPI, PDF uses 72 DPI
            scale = 72 / DPI
            
            logger.info("  ✍️ Overlaying text on PDF...")
            
            # Map corrected lines to OCR bounding boxes
            for idx, block in enumerate(text_blocks):
                # Use corrected line if available, otherwise original OCR
                text_to_insert = corrected_lines[idx] if idx < len(corrected_lines) else block.text
                
                # Scale bbox to PDF coordinates
                x0 = block.bbox[0] * scale
                y0 = block.bbox[1] * scale
                x1 = block.bbox[2] * scale
                y1 = block.bbox[3] * scale
                pdf_rect = fitz.Rect(x0, y0, x1, y1)
                
                # Calculate font size based on box height
                box_height = max(1.0, (y1 - y0))
                font_size = max(4.0, min(box_height * 0.75, 18.0))
                
                try:
                    new_page.insert_textbox(
                        pdf_rect,
                        text_to_insert,
                        fontsize=font_size,
                        fontname="helv",
                        color=(0, 0, 0),
                        render_mode=3,  # Invisible (for searchable PDF)
                        align=0,  # Left align
                    )
                except Exception as e:
                    logger.debug(f"    Text insert warning: {e}")
            
            page_elapsed = time.time() - page_start
            logger.info(f"  ⏱️ Page {page_num + 1} completed in {page_elapsed:.2f}s")
        
        # Save with maximum compression
        logger.info("\n💾 Saving with compression...")
        out_doc.save(
            str(output_path),
            garbage=4,          # Maximum garbage collection
            deflate=True,       # Compress streams
            deflate_images=True,
            deflate_fonts=True,
            clean=True,
        )
        
        out_doc.close()
        src_doc.close()
        
        # Report metrics
        orig_size = input_path.stat().st_size / 1024 / 1024
        new_size = output_path.stat().st_size / 1024 / 1024
        total_elapsed = time.time() - total_start
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Original size:  {orig_size:.2f} MB")
        logger.info(f"  Output size:    {new_size:.2f} MB")
        logger.info(f"  Compression:    {((orig_size - new_size) / orig_size * 100):.1f}%")
        logger.info(f"  Total time:     {total_elapsed:.2f}s")
        logger.info(f"  Avg per page:   {total_elapsed / total_pages:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN
# ============================================================================

def check_gpu_availability():
    """Check and report GPU availability."""
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            gpu_count = paddle.device.cuda.device_count()
            gpu_name = "Unknown"
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_name = result.stdout.strip().split('\n')[0]
            except:
                pass
            logger.info(f"🎮 GPU Available: {gpu_name} (x{gpu_count})")
            return True
        else:
            logger.warning("⚠️ PaddlePaddle not compiled with CUDA")
            return False
    except ImportError:
        logger.warning("⚠️ PaddlePaddle not installed")
        return False

def main():
    """Main entry point."""
    logger.info("\n" + "="*60)
    logger.info("🚀 PDF OCR + VLM Correction Pipeline (Optimized)")
    logger.info("="*60)
    
    # Check GPU
    gpu_available = check_gpu_availability()
    global USE_GPU
    if not gpu_available:
        logger.warning("Running in CPU mode (slower)")
        USE_GPU = False
    
    # Validate paths
    if not INPUT_DIR.exists():
        logger.error(f"❌ Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check VLM server
    logger.info("\n🔌 Checking VLM server connection...")
    try:
        r = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if r.status_code == 200:
            models = r.json().get("data", [])
            logger.info(f"✅ VLM server connected")
            logger.info(f"   Model: {[m.get('id') for m in models]}")
            Config.enable_ai = True
        else:
            logger.error(f"❌ VLM server error (status={r.status_code})")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ VLM server not reachable at {API_BASE_URL}")
        logger.error("   Start it with: ./start_vlm_server.sh")
        sys.exit(1)
    
    # Initialize OCR
    logger.info("\n📦 Initializing OCR engine...")
    ocr_engine = init_paddleocr()
    
    # Find PDFs
    pdf_files = list(INPUT_DIR.rglob("*.pdf"))
    logger.info(f"\n📚 Found {len(pdf_files)} PDF file(s) to process")
    
    if not pdf_files:
        logger.warning("No PDF files found!")
        return
    
    # Process each PDF
    success = 0
    failed = 0
    
    for pdf_path in pdf_files:
        logger.info(f"\n{'#'*60}")
        logger.info(f"📂 File: {pdf_path.name}")
        logger.info(f"{'#'*60}")
        
        # Maintain directory structure in output
        rel_path = pdf_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if create_searchable_pdf(pdf_path, output_path, ocr_engine):
            success += 1
            logger.info(f"✅ Saved: {output_path}")
        else:
            failed += 1
            logger.error(f"❌ Failed: {pdf_path.name}")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("🏁 ALL PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  ✅ Success: {success}")
    logger.info(f"  ❌ Failed:  {failed}")
    logger.info(f"  📁 Output:  {OUTPUT_DIR}")
    logger.info(f"  📊 Diffs:   {DIFF_LOG_DIR}")

if __name__ == "__main__":
    main()