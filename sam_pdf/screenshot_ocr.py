#!/usr/bin/env python3
"""
Screenshot OCR Tool - Win+Shift+E Hotkey
=========================================
When triggered, captures a selected screen area, extracts text via OCR,
and embeds it as a searchable text strip at the bottom of the image.

Usage:
    python screenshot_ocr.py

Dependencies:
    pip install pynput pillow paddleocr opencv-python numpy

The script runs as a background daemon listening for Win+Shift+E.
Screenshots are saved to ~/Pictures/Screenshots_OCR/
"""

import os
import sys
import time
import logging
import subprocess
import tempfile
import textwrap
from pathlib import Path
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pynput import keyboard

# --- CONFIGURATION ---
SCREENSHOT_DIR = Path.home() / "Pictures" / "Screenshots_OCR"
HOTKEY_COMBINATION = {keyboard.Key.cmd, keyboard.Key.shift}  # Win+Shift
HOTKEY_KEY = 'e'  # The letter key to trigger

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# Global state for hotkey detection
current_keys = set()
ocr_engine = None


def init_ocr():
    """Initialize PaddleOCR engine (lazy load for faster startup)"""
    global ocr_engine
    if ocr_engine is None:
        logging.info("🔄 Initializing PaddleOCR (first run may take a moment)...")
        from paddleocr import PaddleOCR
        # Note: use_gpu is auto-detected, use_textline_orientation replaces deprecated use_angle_cls
        ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')
        logging.info("✅ PaddleOCR ready!")
    return ocr_engine


def capture_screen_area():
    """
    Use gnome-screenshot or scrot to capture a selected area.
    Returns the path to the temporary screenshot file, or None if cancelled.
    """
    # Create a temp path (don't create the file - let screenshot tool create it)
    temp_path = tempfile.mktemp(suffix='.png')
    
    # Try gnome-screenshot first (common on GNOME/Pop OS)
    try:
        logging.debug(f"Trying gnome-screenshot, output: {temp_path}")
        result = subprocess.run(
            ['gnome-screenshot', '-a', '-f', temp_path],
            capture_output=True,
            timeout=60  # 60 second timeout for selection
        )
        logging.debug(f"gnome-screenshot returned: {result.returncode}, stderr: {result.stderr.decode() if result.stderr else 'none'}")
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            logging.info(f"✅ Screenshot captured via gnome-screenshot ({os.path.getsize(temp_path)} bytes)")
            return temp_path
        elif result.returncode != 0:
            logging.debug(f"gnome-screenshot failed with code {result.returncode}")
    except FileNotFoundError:
        logging.debug("gnome-screenshot not found")
    except subprocess.TimeoutExpired:
        logging.warning("gnome-screenshot timed out")
    
    # Fallback to scrot
    try:
        logging.debug("Trying scrot...")
        result = subprocess.run(
            ['scrot', '-s', temp_path],
            capture_output=True,
            timeout=60
        )
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            logging.info(f"✅ Screenshot captured via scrot ({os.path.getsize(temp_path)} bytes)")
            return temp_path
    except FileNotFoundError:
        logging.debug("scrot not found")
    except subprocess.TimeoutExpired:
        logging.warning("scrot timed out")
    
    # Fallback to maim (another screenshot tool)
    try:
        logging.debug("Trying maim...")
        result = subprocess.run(
            ['maim', '-s', temp_path],
            capture_output=True,
            timeout=60
        )
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            logging.info(f"✅ Screenshot captured via maim ({os.path.getsize(temp_path)} bytes)")
            return temp_path
    except FileNotFoundError:
        logging.debug("maim not found")
    except subprocess.TimeoutExpired:
        logging.warning("maim timed out")
    
    # Cleanup temp file if capture failed
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    logging.error("❌ Screenshot capture failed (gnome-screenshot may have been cancelled)")
    return None


def process_screenshot(img_path: str, output_path: Path):
    """
    Process the screenshot: run OCR and add text strip at the bottom.
    This makes the image searchable in Samsung Notes and other apps.
    """
    t0 = time.time()
    
    try:
        # Initialize OCR if needed
        ocr = init_ocr()
        
        # 1. Read Image
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            logging.error(f"❌ Could not read image: {img_path}")
            return False

        # 2. Run PaddleOCR (use predict for newer versions)
        logging.info("🔍 Running OCR...")
        result = ocr.predict(img_cv)
        
        extracted_text = []
        # Handle different result formats from PaddleOCR
        if result:
            # New format: result is a list containing dict with 'rec_texts' key
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'rec_texts' in result[0]:
                    # New PaddleOCR format
                    extracted_text = result[0].get('rec_texts', [])
                elif isinstance(result[0], list):
                    # Old format: result[0] is a list of [box, (text, confidence)]
                    for line in result[0]:
                        if len(line) >= 2 and isinstance(line[1], tuple):
                            text = line[1][0]
                            extracted_text.append(text)
        
        logging.debug(f"OCR extracted {len(extracted_text)} text blocks: {extracted_text[:3]}...")
        full_text_str = " ".join(extracted_text)
        
        # 3. Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        width, height = pil_img.size
        
        # If no text found, save the original image
        if not full_text_str.strip():
            pil_img.save(output_path, "PNG")
            logging.warning("⚠️ No text found in screenshot - saved without text strip")
            return True
        
        logging.info(f"📝 Extracted {len(extracted_text)} text blocks")

        # 4. Create Text Strip using Pillow
        # Dynamic font size (2% of width, minimum 12px)
        font_size = max(12, int(width * 0.02))
        
        try:
            # Try common Linux fonts
            font = None
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                "arial.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
            ]
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except IOError:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Wrap text to fit image width
        avg_char_width = font_size * 0.6
        chars_per_line = max(20, int(width / avg_char_width))
        wrapped_lines = textwrap.wrap(full_text_str, width=chars_per_line)
        
        # Calculate footer height
        line_height = font_size + 4
        footer_height = line_height * len(wrapped_lines) + 10
        
        # Create new image with extra space at bottom for text
        new_img = Image.new('RGB', (width, height + footer_height), (255, 255, 255))
        new_img.paste(pil_img, (0, 0))
        
        # 5. Write Text into the Footer
        draw = ImageDraw.Draw(new_img)
        text_y = height + 5
        
        # Use light grey text (visible but not distracting)
        # Samsung Notes will still be able to search this text
        text_color = (100, 100, 100)
        
        for line in wrapped_lines:
            draw.text((10, text_y), line, font=font, fill=text_color)
            text_y += line_height

        # 6. Save as PNG
        new_img.save(output_path, "PNG")
        
        elapsed = time.time() - t0
        logging.info(f"✅ Screenshot saved: {output_path.name} ({elapsed:.2f}s, {len(extracted_text)} text blocks)")
        
        return True

    except Exception as e:
        logging.error(f"❌ Error processing screenshot: {e}")
        return False


def take_screenshot():
    """Main function to capture and process a screenshot."""
    logging.info("📸 Screenshot hotkey triggered! Select an area...")
    
    # Ensure output directory exists
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Capture screen area
    temp_path = capture_screen_area()
    
    if temp_path is None:
        logging.error("❌ Screenshot capture cancelled or failed")
        try:
            subprocess.run([
                'notify-send',
                '❌ Screenshot Failed',
                'Could not capture screen area',
                '-i', 'dialog-error',
                '-t', '3000'
            ], capture_output=True)
        except FileNotFoundError:
            pass
        return
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
    
    # STEP 1: Save the raw screenshot FIRST (so it's never lost)
    try:
        import shutil
        shutil.copy2(temp_path, output_path)
        logging.info(f"💾 Screenshot saved: {output_path.name}")
        
        # Notify user that screenshot is saved
        try:
            subprocess.run([
                'notify-send',
                '📸 Screenshot Saved',
                f'Saved to: {output_path.name}\n🔄 Running OCR...',
                '-i', 'camera-photo',
                '-t', '2000'
            ], capture_output=True)
        except FileNotFoundError:
            pass
            
    except Exception as e:
        logging.error(f"❌ Failed to save screenshot: {e}")
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return
    
    # STEP 2: Run OCR and update the file with text strip
    logging.info("🔍 Starting OCR processing...")
    success = process_screenshot(temp_path, output_path)
    
    # Cleanup temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass
    
    # STEP 3: Send final notification
    if success:
        logging.info(f"✅ OCR complete! Screenshot with text strip: {output_path}")
        try:
            subprocess.run([
                'notify-send',
                '✅ OCR Complete!',
                f'Screenshot ready: {output_path.name}\nText embedded for search.',
                '-i', 'emblem-ok-symbolic',
                '-t', '3000'
            ], capture_output=True)
        except FileNotFoundError:
            pass
    else:
        logging.warning(f"⚠️ OCR failed, but raw screenshot is saved: {output_path}")
        try:
            subprocess.run([
                'notify-send',
                '⚠️ OCR Failed',
                f'Raw screenshot saved: {output_path.name}\n(No text strip added)',
                '-i', 'dialog-warning',
                '-t', '3000'
            ], capture_output=True)
        except FileNotFoundError:
            pass


def on_press(key):
    """Handle key press events."""
    global current_keys
    
    # Add key to current set
    if hasattr(key, 'char'):
        current_keys.add(key.char.lower() if key.char else None)
    else:
        current_keys.add(key)
    
    # Check if hotkey combination is pressed (Win+Shift+E)
    if HOTKEY_COMBINATION.issubset(current_keys):
        if HOTKEY_KEY in current_keys:
            # Run screenshot in a separate thread to not block the listener
            current_keys.clear()  # Reset to prevent multiple triggers
            thread = Thread(target=take_screenshot, daemon=True)
            thread.start()


def on_release(key):
    """Handle key release events."""
    global current_keys
    
    # Remove key from current set
    try:
        if hasattr(key, 'char'):
            current_keys.discard(key.char.lower() if key.char else None)
        else:
            current_keys.discard(key)
    except Exception:
        pass


def main():
    """Main entry point - start the hotkey listener."""
    print("=" * 60)
    print("  📸 Screenshot OCR Tool")
    print("=" * 60)
    print(f"  Hotkey: Win + Shift + E")
    print(f"  Output: {SCREENSHOT_DIR}")
    print("=" * 60)
    print("  Press Ctrl+C to exit")
    print("=" * 60)
    
    # Create output directory
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pre-initialize OCR in background for faster first screenshot
    def preload_ocr():
        try:
            init_ocr()
        except Exception as e:
            logging.warning(f"OCR preload failed (will retry on first use): {e}")
    
    preload_thread = Thread(target=preload_ocr, daemon=True)
    preload_thread.start()
    
    # Start keyboard listener
    logging.info("🎧 Listening for Win+Shift+E...")
    
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
