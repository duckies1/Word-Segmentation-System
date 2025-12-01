# app.py

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import subprocess
import sys
import base64
import io
from PIL import Image
import time

from CCA import extract_text_components_with_rotation  # your CCA segmentation function

# Optional PDF handling: pdf2image (requires poppler).
# To enable PDF -> image conversion:
#   pip install pdf2image
# and install poppler:
# - Windows: download from https://github.com/oschwartz10612/poppler-windows/releases and add bin path to PATH
# - Linux: apt-get install poppler-utils
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

PDF2IMAGE_AVAILABLE = False

app = Flask(__name__, template_folder="templates")

SAVE_PATH = r"E:\Programming\Projects\Para splitter\saved\words.png"
TEMP_DIR = r"E:\Programming\DL\Project - Handwriting Detection\app\Handwriting detector\segments"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)  

# ----------------- UTILS -----------------
def save_canvas_to_path(data_url: str, out_path: str):
    """Decode a canvas dataURL and save as an RGB image with white background."""
    image_data = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    background.save(out_path)
    return out_path

def binarize_for_segmentation(img_gray):
    """Given grayscale image (numpy uint8), return binarized & optionally dilated image for segmentation."""

    val = np.average(img_gray)
    start = time.time()
    img_gray = cv2.fastNlMeansDenoising(img_gray, val, 20, 7, 21)
    gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 20)
    print(f"Time Taken for resize + thresholding: {time.time() - start}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))  # wider horizontally
    img_dilated = cv2.dilate(gray, kernel, iterations=1)
    return img_dilated

def process(input_img_path, row_divisions=10, base_radius=1):
    """
    Run the full pipeline on an image file:
      - read grayscale -> binarize -> segment -> run HTR on each word
    Returns: dict with words list, paragraph string and debug raw logs (concatenated).
    """
    # read
    img_gray = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return {"error": "Could not read image file."}
    img_gray = cv2.resize(img_gray, (0,0), fx=0.4, fy=0.4)

    # binarize for segmentation
    img_dilated = binarize_for_segmentation(img_gray)

    # decide row_height based on image height (same logic as before)
    row_height = max(20, int(img_dilated.shape[0] / row_divisions))

    # Segment into word bounding boxes
    start = time.time()
    boxes, _ = extract_text_components_with_rotation(img_dilated, row_height=row_height, base_radius=base_radius)
    print(f"Segmentation found {len(boxes)} boxes in {time.time() - start:.2f}s")

    # visualization (image with drawn boxes) encoded as a data URL so the
    # frontend can display segmentation results.
    words_out, results = [], []
    vis = cv2.cvtColor(img_gray.copy(), cv2.COLOR_GRAY2BGR)
    for idx, (_, x, y, w, h) in enumerate(boxes, 1):
        # Crop the word image and save for inspection (optional)
        word_img = img_dilated[y:y+h, x:x+w]
        word_img = np.pad(word_img, ((10, 10), (7, 7)), mode='constant', constant_values=255)
        word_path = os.path.join(TEMP_DIR, f"word_{idx}.png")
        try:
            cv2.imwrite(word_path, word_img)
        except Exception:
            pass

        # draw rectangle on visualization
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(idx), (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # placeholder entry for each detected word
        results.append({"id": idx, "box": [int(x), int(y), int(w), int(h)], "word": ""})

    # encode visualization to base64 data URL
    try:
        import io as _io
        _, png = cv2.imencode('.png', vis)
        b64 = base64.b64encode(png.tobytes()).decode('ascii')
        data_url = f"data:image/png;base64,{b64}"
    except Exception:
        data_url = None

    paragraph = ""  # no OCR performed
    return {"words": results, "paragraph": paragraph, "visualization": data_url, "boxes": boxes}

# ----------------- ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_document():
    """
    Accepts multipart/form-data with a 'file' field. Supported:
      - images: png, jpg, jpeg, bmp, tiff
      - pdf: first page will be converted to image if pdf2image + poppler are available
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename
    fname_lower = filename.lower()
    save_path = os.path.join(TEMP_DIR, f"upload_{os.getpid()}_{filename}")
    file.save(save_path)

    # If PDF, convert first page to image (requires pdf2image)
    if fname_lower.endswith(".pdf"):
        if not PDF2IMAGE_AVAILABLE:
            return jsonify({"error": "PDF support requires pdf2image and poppler. Install pdf2image and poppler."}), 500
        try:
            # Convert first page to PIL image
            pages = convert_from_path(save_path, dpi=200, first_page=1, last_page=1)
            if len(pages) == 0:
                return jsonify({"error": "PDF conversion failed (no pages)"}), 500
            pil_img = pages[0].convert("RGB")
            img_path = os.path.join(TEMP_DIR, f"upload_{os.getpid()}_page0.png")
            pil_img.save(img_path)
        except Exception as e:
            return jsonify({"error": f"PDF conversion failed: {e}"}), 500
    else:
        # assume image file -> standardize to PNG (read via PIL to be robust)
        try:
            pil_img = Image.open(save_path).convert("RGB")
            img_path = os.path.join(TEMP_DIR, f"upload_{os.getpid()}_img.png")
            pil_img.save(img_path)
        except Exception as e:
            return jsonify({"error": f"Could not read uploaded image: {e}"}), 400

    # process the image path
    result = process(img_path, row_divisions=10, base_radius=11)
    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict_paragraph():
    """Save a canvas dataURL to `SAVE_PATH` and run the same pipeline as upload."""
    data = None
    try:
        data = request.json.get("image")
    except Exception:
        pass
    if not data:
        return jsonify({"error": "No image provided"}), 400

    # Save the canvas to the configured SAVE_PATH
    try:
        save_canvas_to_path(data, SAVE_PATH)
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {e}"}), 500

    # Run the same processing pipeline as the upload endpoint
    result = process(SAVE_PATH, row_divisions=10, base_radius=11)
    return jsonify(result)

@app.route("/save", methods=["POST"])
def save_image():
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "No image provided"}), 400
    try:
        save_canvas_to_path(data, SAVE_PATH)
        return jsonify({"message": f"Image saved at {SAVE_PATH}"})
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
