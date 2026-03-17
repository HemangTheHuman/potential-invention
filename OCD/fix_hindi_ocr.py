import os
import json
from copy import deepcopy

import cv2
import fitz
import numpy as np
from tqdm import tqdm
import easyocr
import torch


# =========================================================
# CONFIG
# =========================================================

PDF_PATH = "/home/azureuser/kaithi/OCD/data/agni-puran.pdf"
INPUT_JSON = "/home/azureuser/kaithi/OCD/data/hindi.json"
OUTPUT_JSON = "/home/azureuser/kaithi/OCD/data/agni-puran.json"

# Render bigger because your boxes are tiny
PDF_ZOOM = 5.0

# Expand boxes a bit to preserve matras/top line
PAD_X = 6
PAD_Y = 6

# EasyOCR
LANG_LIST = ["hi", "en"]
GPU = True
BATCH_SIZE = 256        # try 128 / 256 / 512 depending on GPU memory
WORKERS = 0             # keep 0 when using GPU to avoid dataloader overhead
DETAIL = 1

# If True, only fix likely-bad words to save time
ONLY_FIX_SUSPECT = False
CONF_THRESHOLD = 0.85

# Optional
SAVE_DEBUG_IMAGES = False
DEBUG_DIR = "/home/azureuser/kaithi/OCD/data/easyocr_debug"


# =========================================================
# UTILS
# =========================================================

def clean_text(text):
    if text is None:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split()).strip()


def has_devanagari(text):
    return any('\u0900' <= ch <= '\u097F' for ch in text)


def suspicious_word(word_obj):
    text = clean_text(word_obj.get("text", ""))
    conf = float(word_obj.get("confidence", 1.0))

    if conf < CONF_THRESHOLD:
        return True
    if not text:
        return True
    if not has_devanagari(text):
        return True
    if len(text) <= 2 and not has_devanagari(text):
        return True

    ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
    if ascii_ratio > 0.8:
        return True

    return False


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


# =========================================================
# JSON WORD DISCOVERY
# =========================================================

def find_all_word_objects(obj, found=None):
    """
    Recursively find dicts matching your word schema:
      - text
      - bbox_pixels OR geometry_normalized
    """
    if found is None:
        found = []

    if isinstance(obj, dict):
        if "text" in obj and ("bbox_pixels" in obj or "geometry_normalized" in obj):
            found.append(obj)

        for v in obj.values():
            find_all_word_objects(v, found)

    elif isinstance(obj, list):
        for item in obj:
            find_all_word_objects(item, found)

    return found


def get_page_index(word_obj):
    for key in ["page_idx", "page_index", "page", "page_no", "page_number"]:
        if key in word_obj:
            try:
                return int(word_obj[key])
            except Exception:
                pass

    # infer from id like p0_b0_l0_w0
    wid = word_obj.get("id", "")
    if wid.startswith("p"):
        try:
            return int(wid.split("_")[0][1:])
        except Exception:
            pass

    return 0


# =========================================================
# PDF RENDER
# =========================================================

def render_single_page(pdf_path, page_idx, zoom=5.0):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    doc.close()
    return img


# =========================================================
# BOX CONVERSION
# =========================================================

def get_bbox_pixels(word_obj, page_shape):
    """
    Prefer bbox_pixels.
    Fallback to geometry_normalized.
    Returns x0, y0, x1, y1 in *original JSON coordinate space*.
    """
    h, w = page_shape[:2]

    if "bbox_pixels" in word_obj:
        bb = word_obj["bbox_pixels"]
        return int(bb["xmin"]), int(bb["ymin"]), int(bb["xmax"]), int(bb["ymax"])

    geom = word_obj.get("geometry_normalized")
    if geom and isinstance(geom, list) and len(geom) >= 4:
        xs = [pt[0] for pt in geom]
        ys = [pt[1] for pt in geom]
        return int(min(xs) * w), int(min(ys) * h), int(max(xs) * w), int(max(ys) * h)

    return None


def scale_bbox_for_rendered_page(word_obj, rendered_shape):
    """
    Your bbox_pixels appear to come from the original page image size.
    geometry_normalized is safer because it scales automatically.

    If geometry_normalized exists, use it to derive coordinates directly on rendered page.
    Otherwise assume bbox_pixels came from some base page size and approximate using
    rendered size only if original_page_size fields exist.
    """
    h, w = rendered_shape[:2]

    geom = word_obj.get("geometry_normalized")
    if geom and isinstance(geom, list) and len(geom) >= 4:
        xs = [pt[0] for pt in geom]
        ys = [pt[1] for pt in geom]
        x0 = int(min(xs) * w)
        y0 = int(min(ys) * h)
        x1 = int(max(xs) * w)
        y1 = int(max(ys) * h)
        return x0, y0, x1, y1

    # fallback to bbox_pixels if normalized geom is not available
    if "bbox_pixels" in word_obj:
        bb = word_obj["bbox_pixels"]
        x0 = int(bb["xmin"])
        y0 = int(bb["ymin"])
        x1 = int(bb["xmax"])
        y1 = int(bb["ymax"])
        return x0, y0, x1, y1

    return None


# =========================================================
# MAIN PAGE-WISE GPU BATCH OCR
# =========================================================

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if SAVE_DEBUG_IMAGES:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed = deepcopy(data)

    print("Finding word objects...")
    words = find_all_word_objects(fixed)
    print(f"Found {len(words)} candidate word objects")

    # group words by page
    page_to_entries = {}
    for idx, word_obj in enumerate(words):
        page_idx = get_page_index(word_obj)
        page_to_entries.setdefault(page_idx, []).append((idx, word_obj))

    print("Loading EasyOCR reader...")
    # detector=False because we already have boxes
    reader = easyocr.Reader(
        LANG_LIST,
        gpu=GPU,
        detector=False,
        recognizer=True,
        verbose=True,
    )

    total_processed = 0
    total_changed = 0

    page_indices = sorted(page_to_entries.keys())

    for page_idx in tqdm(page_indices, desc="Processing pages"):
        entries = page_to_entries[page_idx]

        page_img_bgr = render_single_page(PDF_PATH, page_idx, zoom=PDF_ZOOM)
        page_img_rgb = cv2.cvtColor(page_img_bgr, cv2.COLOR_BGR2RGB)
        h, w = page_img_rgb.shape[:2]

        horizontal_list = []
        mapping = []

        # Build batch of known word boxes for this page
        for global_idx, word_obj in entries:
            if ONLY_FIX_SUSPECT and not suspicious_word(word_obj):
                continue

            bbox = scale_bbox_for_rendered_page(word_obj, page_img_rgb.shape)
            if bbox is None:
                continue

            x0, y0, x1, y1 = bbox
            x0 = clamp(x0 - PAD_X, 0, w - 1)
            y0 = clamp(y0 - PAD_Y, 0, h - 1)
            x1 = clamp(x1 + PAD_X, 0, w - 1)
            y1 = clamp(y1 + PAD_Y, 0, h - 1)

            if x1 <= x0 or y1 <= y0:
                continue

            # EasyOCR recognize() expects [x_min, x_max, y_min, y_max]
            horizontal_list.append([x0, x1, y0, y1])
            mapping.append(global_idx)

        if not horizontal_list:
            continue

        # Batched GPU recognition on provided boxes
        # This skips detection and only recognizes your word crops.
        try:
            results = reader.recognize(
                page_img_rgb,
                horizontal_list=horizontal_list,
                free_list=[],
                decoder="greedy",
                beamWidth=5,
                batch_size=BATCH_SIZE,
                workers=WORKERS,
                allowlist=None,
                blocklist=None,
                detail=DETAIL,
                paragraph=False,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                filter_ths=0.003,
                width_ths=0.0,
                height_ths=0.0,
                ycenter_ths=0.0,
                x_ths=0.0,
                y_ths=0.0,
            )
        except TypeError:
            # Some EasyOCR versions expose a slightly smaller recognize signature
            results = reader.recognize(
                page_img_rgb,
                horizontal_list=horizontal_list,
                free_list=[],
                decoder="greedy",
                beamWidth=5,
                batch_size=BATCH_SIZE,
                workers=WORKERS,
                detail=DETAIL,
                paragraph=False,
                contrast_ths=0.05,
                adjust_contrast=0.7,
            )

        # results should align with horizontal_list order
        for map_idx, result in enumerate(results):
            word_obj = words[mapping[map_idx]]
            old_text = clean_text(word_obj.get("text", ""))

            if isinstance(result, (list, tuple)) and len(result) >= 3:
                # usually: [box, text, confidence]
                new_text = clean_text(result[1])
                new_conf = float(result[2]) if result[2] is not None else None
            elif isinstance(result, (list, tuple)) and len(result) == 2:
                new_text = clean_text(result[0])
                new_conf = float(result[1]) if result[1] is not None else None
            else:
                new_text = clean_text(str(result))
                new_conf = None

            total_processed += 1

            if new_text and new_text != old_text:
                word_obj["text"] = new_text
                total_changed += 1

            word_obj["ocr_fix_meta"] = {
                "old_text": old_text,
                "new_text": new_text,
                "new_confidence": new_conf,
                "method": "easyocr_recognize_gpu_batched_word_level",
                "batch_size": BATCH_SIZE,
                "pdf_zoom": PDF_ZOOM,
            }

        if SAVE_DEBUG_IMAGES and page_idx < 3:
            dbg = page_img_bgr.copy()
            for box in horizontal_list[:100]:
                x0, x1, y0, y1 = box
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"page_{page_idx}_boxes.jpg"), dbg)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Total re-recognized words: {total_processed}")
    print(f"Changed words: {total_changed}")
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()