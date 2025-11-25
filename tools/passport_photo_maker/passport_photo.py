import cv2
import os
from PIL import Image

def process_passport_photo(input_path, output_path, cascade_path):
    """
    Processes an image to meet passport photo specifications by detecting a face,
    cropping it to the correct aspect ratio and prominence, and resizing it.

    Args:
        input_path (str): Path to the source image.
        output_path (str): Path to save the processed image.
        cascade_path (str): Path to the Haar Cascade XML file for face detection.
    """
    # --- Configuration ---
    TARGET_WIDTH_CM = 3.5
    TARGET_HEIGHT_CM = 4.5
    DPI = 300
    FACE_HEIGHT_PERCENTAGE = 0.80

    # --- Calculations ---
    target_width_px = int(TARGET_WIDTH_CM / 2.54 * DPI)
    target_height_px = int(TARGET_HEIGHT_CM / 2.54 * DPI)
    aspect_ratio = TARGET_WIDTH_CM / TARGET_HEIGHT_CM

    # --- Face Detection ---
    if not os.path.exists(cascade_path):
        print(f"❌ Error: Cascade file not found at {cascade_path}")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)

    image_cv = cv2.imread(input_path)
    if image_cv is None:
        print(f"❌ Error: Could not read the image at {input_path}. Check the file path and format.")
        return

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Adjusted minSize for better detection on smaller images
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        print("❌ No face detected. Please use a clearer, front-facing photo.")
        return
    elif len(faces) > 1:
        print("⚠️ Multiple faces detected. Using the largest one.")
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    x, y, w, h = faces[0]

    # --- Cropping Logic ---
    crop_height = int(h / FACE_HEIGHT_PERCENTAGE)
    crop_width = int(crop_height * aspect_ratio)
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    crop_x1 = face_center_x - crop_width // 2
    crop_y1 = face_center_y - crop_height // 2

    # Boundary checks to ensure the crop box is within the image
    img_h, img_w, _ = image_cv.shape
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x1 + crop_width)
    crop_y2 = min(img_h, crop_y1 + crop_height)
    crop_x1 = max(0, crop_x2 - crop_width)
    crop_y1 = max(0, crop_y2 - crop_height)

    # --- Final Processing using Pillow ---
    pil_img = Image.open(input_path)
    cropped_img = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Handle different Pillow versions for resizing
    try:
        # For modern Pillow (8.0.0+)
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        # For older Pillow
        resample_filter = Image.LANCZOS

    final_img = cropped_img.resize((target_width_px, target_height_px), resample_filter)

    # --- Saving Logic ---
    output_format = 'JPEG'
    file_ext = os.path.splitext(output_path)[1].lower()
    if file_ext == '.png':
        output_format = 'PNG'
    
    print(f"Saving image as {output_format}...")
    try:
        if output_format == 'JPEG':
            if final_img.mode == 'RGBA':
                final_img = final_img.convert('RGB')
            final_img.save(output_path, output_format, quality=95, dpi=(DPI, DPI))
        else: # For PNG
            final_img.save(output_path, output_format, dpi=(DPI, DPI))
            
    except OSError:
        print("⚠️ Could not save with DPI metadata due to a system library issue.")
        print("   Saving again without DPI information.")
        if output_format == 'JPEG':
            final_img.save(output_path, output_format, quality=95)
        else:
            final_img.save(output_path, output_format)

    print(f"✅ Successfully processed and saved image to {output_path}")
    print(f"   Final dimensions: {target_width_px} x {target_height_px} pixels.")


# --- How to Use ---
if __name__ == "__main__":
    # ✍️ 1. Set the path to your input image file
    input_filename = "/home/pope/Desktop/veena.jpeg"
    
    # ✍️ 2. Set the path for your saved output file (e.g., passport.jpg or passport.png)
    output_filename = "passport_photo_final.jpg"
    
    # -------------------------------------------------------------
    
    # This automatically finds the face detection file
    cascade_file_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

    if not os.path.exists(input_filename):
        print(f"❌ Error: Input file not found at '{input_filename}'")
    else:
        process_passport_photo(input_filename, output_filename, cascade_file_path)