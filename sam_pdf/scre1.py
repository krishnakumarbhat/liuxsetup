import os
import time
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def setup_args():
    parser = argparse.ArgumentParser(description="Add OCR Text Strip to Images")
    parser.add_argument("input_dir", type=str, help="Directory with Screenshots")
    parser.add_argument("output_dir", type=str, help="Where to save processed Images")
    return parser.parse_args()

def process_image(img_path, output_path, ocr):
    t0 = time.time()
    
    try:
        # 1. Read Image
        # We use cv2 for OCR reading because it handles formats well
        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            return

        # 2. Run PaddleOCR
        result = ocr.ocr(img_cv, cls=True)
        
        extracted_text = []
        if result and result[0]:
            # Gather all text found in the image
            for line in result[0]:
                text = line[1][0]
                extracted_text.append(text)
        
        full_text_str = " ".join(extracted_text)
        
        # If no text found, just copy the image
        if not full_text_str.strip():
            Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)).save(output_path)
            logging.warning(f"No text found: {img_path.name}")
            return

        # 3. Create Text Strip using Pillow
        # Convert CV2 (BGR) to PIL (RGB)
        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        width, height = pil_img.size
        
        # Calculate needed height for text
        # We'll use a small font size relative to image width
        font_size = max(12, int(width * 0.02)) # Dynamic font size (2% of width)
        try:
            # Try to load a default font, otherwise load default
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Wrap text? For a simple search strip, we can just dump it in lines
        # or just put it all in one block. Let's do a simple wrap.
        avg_char_width = font_size * 0.6
        chars_per_line = int(width / avg_char_width)
        import textwrap
        wrapped_lines = textwrap.wrap(full_text_str, width=chars_per_line)
        
        # Calculate new footer height
        line_height = font_size + 4
        footer_height = line_height * len(wrapped_lines) + 10 # +10 padding
        
        # Create new blank image with extra space at bottom
        new_img = Image.new('RGB', (width, height + footer_height), (255, 255, 255))
        new_img.paste(pil_img, (0, 0))
        
        # 4. Write Text into the Footer
        draw = ImageDraw.Draw(new_img)
        text_y = height + 5
        
        # Color: Very light grey (so it's barely visible to you, but readable by machine)
        # or Black (if you want to read it). Let's do Dark Grey.
        text_color = (80, 80, 80) 
        
        for line in wrapped_lines:
            draw.text((10, text_y), line, font=font, fill=text_color)
            text_y += line_height

        # 5. Save as PNG
        output_path = output_path.with_suffix('.png') # Force PNG
        new_img.save(output_path, "PNG")
        
        elapsed = time.time() - t0
        logging.info(f"✅ Saved: {output_path.name} ({elapsed:.2f}s)")

    except Exception as e:
        logging.error(f"❌ Error on {img_path.name}: {e}")

def main():
    args = setup_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print("Input directory not found.")
        return

    # Initialize PaddleOCR (GPU)
    logging.info("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)

    # Find images
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(list(input_root.rglob(ext)))
    
    logging.info(f"Found {len(images)} images.")

    for img in images:
        rel_path = img.relative_to(input_root)
        dest_path = output_root / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        process_image(img, dest_path, ocr)

if __name__ == "__main__":
    main()