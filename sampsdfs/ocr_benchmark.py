#!/usr/bin/env python3
"""
OCR Benchmarking Script - VLM & Layout
======================================
Engines:
1. Qwen2-VL (Generative VLM - Best for Math)
2. Surya OCR (Layout/Text - "Chandra" alternative)
3. ABBYY FineReader (Commercial CLI)
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Any
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_pdf_to_images(pdf_path: Path, dpi: int = 150) -> List[Any]:
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(pdf_path), dpi=dpi)
        logger.info(f"Converted {len(images)} pages at {dpi} DPI")
        return images
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return []

def save_output(output_dir: Path, name: str, md_content: str):
    md_path = output_dir / f"output_{name.lower()}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    logger.info(f"Saved MD: {md_path}")

# =============================================================================
# 1. QWEN2-VL (Generative VLM)
# =============================================================================

def run_qwen_vl(pdf_path: Path, output_dir: Path) -> bool:
    """Run Qwen2-VL-2B-Instruct."""
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        logger.info(f"Loading {model_id}...")
        
        # Load model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        images = convert_pdf_to_images(pdf_path, dpi=100)
        md_content = ["# Qwen2-VL Output\n"]
        
        for i, img in enumerate(images):
            logger.info(f"Qwen2-VL: Processing page {i+1}")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Transcribe this document into Markdown. Output math in LaTeX format."}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            md_content.append(f"\n## Page {i+1}\n")
            md_content.append(output_text)
            
        save_output(output_dir, "qwen_vl", '\n'.join(md_content))
        return True
    except Exception as e:
        logger.error(f"Qwen2-VL failed: {e}")
        return False

# =============================================================================
# 2. SURYA OCR (Layout/Text)
# =============================================================================

def run_surya(pdf_path: Path, output_dir: Path) -> bool:
    """Run Surya OCR."""
    try:
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        
        logger.info("Loading Surya models...")
        det_processor = load_det_processor()
        det_model = load_det_model()
        rec_model = load_rec_model()
        rec_processor = load_rec_processor()
        
        images = convert_pdf_to_images(pdf_path, dpi=150)
        md_content = ["# Surya OCR Output\n"]
        
        for i, img in enumerate(images):
            logger.info(f"Surya: Processing page {i+1}")
            # Surya expects list of images and list of langs
            predictions = run_ocr([img], [["en"]], det_model, det_processor, rec_model, rec_processor)
            
            md_content.append(f"\n## Page {i+1}\n")
            if predictions and len(predictions) > 0:
                for line in predictions[0].text_lines:
                    md_content.append(line.text)
        
        save_output(output_dir, "surya", '\n'.join(md_content))
        return True
    except Exception as e:
        logger.error(f"Surya failed: {e}")
        return False

# =============================================================================
# 3. ABBYY FINEREADER
# =============================================================================

def run_abbyy(pdf_path: Path, output_dir: Path) -> bool:
    """Run ABBYY FineReader CLI."""
    abbyy_cmd = "abbyyocr11"
    try:
        subprocess.run([abbyy_cmd, "--help"], capture_output=True)
        logger.info("Running ABBYY...")
        output_path = output_dir / "output_abbyy.pdf"
        subprocess.run([abbyy_cmd, "-if", str(pdf_path), "-of", str(output_path), "-f", "PDF"], check=True)
        return True
    except FileNotFoundError:
        logger.warning("ABBYY FineReader CLI not found.")
        save_output(output_dir, "abbyy", "# ABBYY FineReader\n\nSoftware not installed.")
        return False
    except Exception as e:
        logger.error(f"ABBYY failed: {e}")
        return False

def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    pdf_path = Path("calculus.pdf")
    
    print(f"Benchmarking on {pdf_path}...")
    run_surya(pdf_path, output_dir)
    run_qwen_vl(pdf_path, output_dir)
    run_abbyy(pdf_path, output_dir)
    print("\nDone!")

if __name__ == "__main__":
    main()
