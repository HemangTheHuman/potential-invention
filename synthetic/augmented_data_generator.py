import os
import re
import cv2
import glob
import json
import math
import random
import argparse
import multiprocessing as mp
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from aksharamukha import transliterate
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch

try:
    import kornia.filters as KF
except Exception:
    KF = None

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================================================
# USER: REPLACE THIS WITH YOUR REAL TRANSLITERATION FUNCTION
# =========================================================

def devanagari_to_kaithi(text: str) -> str:
    """
    Replace this with your actual transliteration function.
    Example:
        from your_module import transliterate
        return transliterate(text)
    """
    src='Devanagari'
    tgt='Kaithi'
    txt=text.lower()
    res=transliterate.process(src, tgt, txt, nativize =False, pre_options = [], post_options = [])
    return res


# =========================================================
# Helpers
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_box_xyxy(x1, y1, x2, y2, w, h):
    return [[float(x1) / w, float(y1) / h], [float(x2) / w, float(y2) / h]]


def normalize_quad(quad, w, h):
    return [[float(x) / w, float(y) / h] for x, y in quad]


def union_xyxy(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def list_font_files(font_dir):
    files = []
    for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
        files.extend(glob.glob(os.path.join(font_dir, ext)))
    return sorted(files)


def load_font(font_path, size):
    return ImageFont.truetype(font_path, size=size)


# =========================================================
# Corpus loading
# =========================================================

TOKEN_RE = re.compile(r"[^\s]+")


def collect_txt_files(corpus_dir):
    files = []
    for ext in ("*.txt", "*.TXT"):
        files.extend(glob.glob(os.path.join(corpus_dir, "**", ext), recursive=True))
    return sorted(files)


def load_devanagari_tokens(corpus_dir, min_len=1):
    files = collect_txt_files(corpus_dir)
    if not files:
        raise ValueError(f"No txt files found in {corpus_dir}")

    tokens = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for tok in TOKEN_RE.findall(line):
                    tok = tok.strip()
                    if len(tok) >= min_len:
                        tokens.append(tok)

    if not tokens:
        raise ValueError("No usable tokens found in corpus.")
    return tokens


# =========================================================
# Background patches
# =========================================================

def load_patch_index(bg_root):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    categories = ["clean_paper", "stained_paper", "edge_and_shadow"]
    patch_db = {k: [] for k in categories}

    for cat in categories:
        cat_dir = os.path.join(bg_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for ext in exts:
            patch_db[cat].extend(glob.glob(os.path.join(cat_dir, ext)))

    total = sum(len(v) for v in patch_db.values())
    if total == 0:
        raise ValueError(f"No patch images found in {bg_root}")

    return patch_db


def sample_patch_path(patch_db):
    cat = random.choices(
        ["clean_paper", "stained_paper", "edge_and_shadow"],
        weights=[0.45, 0.35, 0.20],
        k=1
    )[0]
    if patch_db[cat]:
        return random.choice(patch_db[cat])
    non_empty = [k for k, v in patch_db.items() if v]
    return random.choice(patch_db[random.choice(non_empty)])


def read_crop_or_tile_patch(img_path, target_h, target_w):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read patch: {img_path}")

    h, w = img.shape[:2]
    if h >= target_h and w >= target_w:
        y = random.randint(0, h - target_h)
        x = random.randint(0, w - target_w)
        return img[y:y+target_h, x:x+target_w].copy()

    # tile small patches
    reps_y = math.ceil(target_h / h)
    reps_x = math.ceil(target_w / w)
    tiled = np.tile(img, (reps_y, reps_x, 1))
    return tiled[:target_h, :target_w].copy()


def make_realistic_background(page_h, page_w, patch_db):
    base_path = sample_patch_path(patch_db)
    bg = read_crop_or_tile_patch(base_path, page_h, page_w)

    # mild extra tone drift
    noise = np.random.normal(0, 4, (page_h, page_w, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=17, sigmaY=17)
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return bg


# =========================================================
# Layout templates
# =========================================================

def draw_page_border(draw, page_w, page_h):
    m = random.randint(24, 42)
    draw.rectangle([m, m, page_w - m, page_h - m], outline=(40, 40, 40), width=random.randint(1, 2))


def draw_table(draw, x, y, w, h, rows, cols, line_color=(55, 55, 55), width=1):
    draw.rectangle([x, y, x + w, y + h], outline=line_color, width=width)
    xs = [x + int(w * c / cols) for c in range(cols)] + [x + w]
    ys = [y + int(h * r / rows) for r in range(rows)] + [y + h]

    for vx in xs[1:-1]:
        draw.line([vx, y, vx, y + h], fill=line_color, width=width)
    for hy in ys[1:-1]:
        draw.line([x, hy, x + w, hy], fill=line_color, width=width)

    return xs, ys


def pick_layout():
    return random.choices(["record", "ledger", "endorsement"], weights=[0.45, 0.40, 0.15], k=1)[0]


def draw_record_template(draw, page_w, page_h, printed_font):
    zones = []

    title = random.choice(["Record of Rights", "Land Record Register", "Village Record", "Settlement Record"])
    draw.text((page_w//2 - 120, random.randint(60, 90)), title, font=printed_font, fill=(30, 30, 30))

    labels = [
        ("Mauza", 70, 135),
        ("Thana", 70, 175),
        ("District", 70, 215),
        ("Number", page_w - 250, 135),
        ("Pargana", page_w - 250, 175),
    ]
    for txt, x, y in labels:
        draw.text((x, y), txt, font=printed_font, fill=(45, 45, 45))
        draw.line((x + 75, y + 18, x + 215, y + 18), fill=(70, 70, 70), width=1)

    table_x, table_y = 55, 290
    table_w, table_h = page_w - 110, int(page_h * 0.42)

    cols = [0.10, 0.47, 0.23, 0.20]
    rows = random.randint(6, 9)

    draw.rectangle([table_x, table_y, table_x + table_w, table_y + table_h], outline=(50, 50, 50), width=1)

    xs = [table_x]
    cur = table_x
    for p in cols[:-1]:
        cur += int(table_w * p)
        xs.append(cur)
    xs.append(table_x + table_w)

    for vx in xs[1:-1]:
        draw.line([vx, table_y, vx, table_y + table_h], fill=(55, 55, 55), width=1)

    row_h = table_h // rows
    ys = [table_y + i * row_h for i in range(rows+1)]
    for hy in ys[1:-1]:
        draw.line([table_x, hy, table_x + table_w, hy], fill=(55, 55, 55), width=1)

    headers = ["Serial", "Description of Papers", "From Which No.", "Remarks"]
    for i in range(4):
        draw.text((xs[i] + 8, ys[0] + 6), headers[i], font=printed_font, fill=(45, 45, 45))

    block_id = 0
    for r in range(1, rows):
        zones.append({"block_id": block_id, "type": "serial", "bbox": [xs[0]+6, ys[r]+6, xs[1]-6, ys[r+1]-6], "max_words": 1}); block_id += 1
        zones.append({"block_id": block_id, "type": "desc", "bbox": [xs[1]+8, ys[r]+6, xs[2]-8, ys[r+1]-6], "max_words": random.randint(2, 8)}); block_id += 1
        zones.append({"block_id": block_id, "type": "ref", "bbox": [xs[2]+8, ys[r]+6, xs[3]-8, ys[r+1]-6], "max_words": random.randint(1, 4)}); block_id += 1
        zones.append({"block_id": block_id, "type": "remark", "bbox": [xs[3]+8, ys[r]+6, xs[4]-8, ys[r+1]-6], "max_words": random.randint(0, 3)}); block_id += 1

    zones.append({
        "block_id": block_id,
        "type": "paragraph",
        "bbox": [70, table_y + table_h + 35, page_w - 80, page_h - 120],
        "max_words": random.randint(18, 38)
    })
    return zones


def draw_ledger_template(draw, page_w, page_h, printed_font):
    zones = []

    title = random.choice(["Ledger Register", "Continuous Sheet", "Landholding Register", "Revenue Sheet"])
    draw.text((page_w//2 - 100, random.randint(50, 80)), title, font=printed_font, fill=(28, 28, 28))

    table_x, table_y = 45, 135
    table_w, table_h = page_w - 90, page_h - 220

    cols = random.randint(7, 10)
    rows = random.randint(12, 18)
    xs, ys = draw_table(draw, table_x, table_y, table_w, table_h, rows, cols, line_color=(60, 60, 60), width=1)

    for c in range(cols):
        txt = random.choice(["No.", "Entry", "Khata", "Area", "Name", "Ref.", "Remark", "Patti", "Jamabandi"])
        draw.text((xs[c] + 4, ys[0] + 4), txt, font=printed_font, fill=(40, 40, 40))

    block_id = 0
    for r in range(1, rows):
        for c in range(cols):
            if random.random() < 0.38:
                zones.append({
                    "block_id": block_id,
                    "type": "cell",
                    "bbox": [xs[c] + 4, ys[r] + 4, xs[c+1] - 4, ys[r+1] - 4],
                    "max_words": random.randint(1, 4)
                })
                block_id += 1

    if random.random() < 0.35:
        zones.append({
            "block_id": block_id,
            "type": "margin_note",
            "bbox": [page_w - 130, table_y + 40, page_w - 50, table_y + 230],
            "max_words": random.randint(4, 10)
        })
    return zones


def draw_endorsement_template(draw, page_w, page_h, printed_font):
    zones = []

    if random.random() < 0.65:
        poly = [
            (0, 0),
            (int(page_w * random.uniform(0.45, 0.72)), 0),
            (page_w, int(page_h * random.uniform(0.18, 0.38))),
            (page_w, 0)
        ]
        draw.polygon(poly, fill=(230, 235, 235))

    title = random.choice(["Register of Mutation", "Sheet of Endorsement", "Settlement Entry", "Change Register"])
    draw.text((60, 50), title, font=printed_font, fill=(30, 30, 30))

    labels = [("Village", 70, 120), ("Khata", 70, 165), ("Sheet", 330, 120), ("Khasra", 330, 165)]
    for txt, x, y in labels:
        draw.text((x, y), txt, font=printed_font, fill=(42, 42, 42))
        draw.line((x + 60, y + 18, x + 200, y + 18), fill=(65, 65, 65), width=1)

    table_x, table_y = 50, 245
    table_w, table_h = page_w - 100, int(page_h * 0.42)

    xs, ys = draw_table(draw, table_x, table_y, table_w, table_h, random.randint(5, 8), 4, line_color=(55, 55, 55), width=1)
    headers = ["Particulars", "Date", "Signature", "Remarks"]
    for i in range(4):
        draw.text((xs[i] + 6, ys[0] + 6), headers[i], font=printed_font, fill=(45, 45, 45))

    block_id = 0
    for r in range(1, len(ys)-1):
        zones.append({"block_id": block_id, "type": "particulars", "bbox": [xs[0]+6, ys[r]+6, xs[1]-6, ys[r+1]-6], "max_words": random.randint(4, 10)}); block_id += 1
        zones.append({"block_id": block_id, "type": "date", "bbox": [xs[1]+6, ys[r]+6, xs[2]-6, ys[r+1]-6], "max_words": random.randint(1, 2)}); block_id += 1
        zones.append({"block_id": block_id, "type": "sig", "bbox": [xs[2]+6, ys[r]+6, xs[3]-6, ys[r+1]-6], "max_words": random.randint(0, 2)}); block_id += 1
        zones.append({"block_id": block_id, "type": "remarks", "bbox": [xs[3]+6, ys[r]+6, xs[4]-6, ys[r+1]-6], "max_words": random.randint(0, 3)}); block_id += 1

    zones.append({
        "block_id": block_id,
        "type": "footer_note",
        "bbox": [70, table_y + table_h + 30, page_w - 80, page_h - 90],
        "max_words": random.randint(12, 24)
    })
    return zones


# =========================================================
# Text rendering
# =========================================================

def random_ink_color():
    return random.choice([
        (20, 20, 20),
        (35, 30, 24),
        (55, 45, 38),
        (28, 26, 42),
        (18, 18, 32),
    ])


def draw_word_with_variation(base_rgba, text, x, y, font, angle_deg=0):
    tmp_draw = ImageDraw.Draw(base_rgba)
    x1, y1, x2, y2 = tmp_draw.textbbox((0, 0), text, font=font)
    tw, th = x2 - x1, y2 - y1

    pad = 8
    word_img = Image.new("RGBA", (tw + 2*pad, th + 2*pad), (255, 255, 255, 0))
    d = ImageDraw.Draw(word_img)
    d.text((pad, pad), text, font=font, fill=random_ink_color() + (255,))

    arr = np.array(word_img)
    alpha = arr[..., 3]

    if random.random() < 0.35:
        alpha = cv2.erode(alpha, np.ones((random.choice([1, 1, 2]), random.choice([1, 1, 2])), np.uint8), iterations=1)
    if random.random() < 0.25:
        alpha = cv2.dilate(alpha, np.ones((random.choice([1, 2]), random.choice([1, 2])), np.uint8), iterations=1)
    if random.random() < 0.22:
        fade = np.random.uniform(0.72, 1.0, alpha.shape).astype(np.float32)
        fade = cv2.GaussianBlur(fade, (0, 0), sigmaX=random.uniform(5, 15))
        alpha = np.clip(alpha.astype(np.float32) * fade, 0, 255).astype(np.uint8)

    arr[..., 3] = alpha
    word_img = Image.fromarray(arr)

    if abs(angle_deg) > 0.1:
        word_img = word_img.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)

    base_rgba.alpha_composite(word_img, dest=(int(x), int(y)))

    wa = np.array(word_img)[..., 3]
    ys, xs = np.where(wa > 10)
    if len(xs) == 0:
        return None

    bx1 = int(x + xs.min())
    by1 = int(y + ys.min())
    bx2 = int(x + xs.max())
    by2 = int(y + ys.max())
    quad = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]

    return {"bbox": [bx1, by1, bx2, by2], "quad": quad}


def fit_font_size_for_zone(pairs, zone_w, zone_h, font_path, min_sz=22, max_sz=38):
    dummy = Image.new("RGB", (zone_w, zone_h), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)

    for sz in range(max_sz, min_sz - 1, -1):
        font = load_font(font_path, sz)
        cursor_x = 0
        cursor_y = 0
        line_h = sz + random.randint(6, 12)
        ok = True

        for _, kai in pairs:
            x1, y1, x2, y2 = draw.textbbox((0, 0), kai, font=font)
            tw = x2 - x1
            if cursor_x + tw > zone_w:
                cursor_x = 0
                cursor_y += line_h
            if cursor_y + line_h > zone_h:
                ok = False
                break
            cursor_x += tw + random.randint(8, 20)

        if ok:
            return font

    return load_font(font_path, min_sz)


# =========================================================
# Augmentations
# =========================================================

def perspective_transform_with_annotations(img, page_items, max_ratio=0.045):
    h, w = img.shape[:2]
    dx = int(w * max_ratio)
    dy = int(h * max_ratio)

    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst = np.float32([
        [random.randint(0, dx), random.randint(0, dy)],
        [w-1-random.randint(0, dx), random.randint(0, dy)],
        [w-1-random.randint(0, dx), h-1-random.randint(0, dy)],
        [random.randint(0, dx), h-1-random.randint(0, dy)],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    new_words = []
    for wd in page_items["words"]:
        quad = np.array(wd["quad"], dtype=np.float32).reshape(-1, 1, 2)
        new_quad = cv2.perspectiveTransform(quad, M).reshape(-1, 2)

        xs = new_quad[:, 0]
        ys = new_quad[:, 1]
        x1 = clamp(int(np.floor(xs.min())), 0, w-1)
        y1 = clamp(int(np.floor(ys.min())), 0, h-1)
        x2 = clamp(int(np.ceil(xs.max())), 0, w-1)
        y2 = clamp(int(np.ceil(ys.max())), 0, h-1)

        if x2 <= x1 or y2 <= y1:
            continue

        item = dict(wd)
        item["quad"] = [[float(x), float(y)] for x, y in new_quad.tolist()]
        item["bbox"] = [x1, y1, x2, y2]
        new_words.append(item)

    page_items["words"] = new_words
    return warped, page_items


def add_stains(img, count_range=(1, 6)):
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    for _ in range(random.randint(*count_range)):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        rx = random.randint(max(15, w // 35), max(25, w // 8))
        ry = random.randint(max(15, h // 35), max(25, h // 8))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=random.uniform(12, 30), sigmaY=random.uniform(12, 30))
        alpha = (mask.astype(np.float32) / 255.0) * random.uniform(0.03, 0.10)

        stain_color = np.array([
            random.randint(85, 145),
            random.randint(100, 160),
            random.randint(125, 190)
        ], dtype=np.float32)

        out = out * (1 - alpha[..., None]) + stain_color * alpha[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def add_fold_lines(img):
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(random.randint(1, 2)):
        if random.random() < 0.55:
            x = random.randint(int(w * 0.15), int(w * 0.85))
            cv2.line(out, (x, 0), (x, h), (150, 150, 150), thickness=random.randint(1, 2))
        else:
            y = random.randint(int(h * 0.15), int(h * 0.85))
            cv2.line(out, (0, y), (w, y), (150, 150, 150), thickness=random.randint(1, 2))
    return cv2.GaussianBlur(out, (0, 0), sigmaX=0.8)


def add_bleed_through(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    bleed = cv2.flip(inv, 1)
    bleed = cv2.GaussianBlur(bleed, (0, 0), sigmaX=7, sigmaY=7)
    bleed3 = cv2.cvtColor(bleed, cv2.COLOR_GRAY2BGR).astype(np.float32)
    alpha = random.uniform(0.02, 0.06)
    out = img.astype(np.float32) - alpha * bleed3
    return np.clip(out, 0, 255).astype(np.uint8)


def add_small_occlusion(img):
    h, w = img.shape[:2]
    out = img.copy()
    poly = np.array([
        [random.randint(0, w // 2), 0],
        [random.randint(w // 3, w - 1), 0],
        [random.randint(w // 2, w - 1), random.randint(h // 8, h // 3)],
        [random.randint(0, w // 3), random.randint(h // 12, h // 3)]
    ], dtype=np.int32)
    color = (random.randint(200, 235), random.randint(210, 240), random.randint(215, 245))
    cv2.fillConvexPoly(out, poly, color)
    return out


def jpeg_compress(img):
    q = random.randint(45, 82)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def gpu_post_augment(img, device):
    if device.type != "cuda" or KF is None:
        return img

    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(x).float().to(device) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)

    _, _, h, w = x.shape

    # illumination
    if random.random() < 0.82:
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing="ij"
        )
        a = random.uniform(-0.16, 0.16)
        b = random.uniform(-0.16, 0.16)
        c = random.uniform(0.90, 1.06)
        light = (c + a * xx + b * yy).unsqueeze(0).unsqueeze(0)
        k = 61 if min(h, w) >= 61 else (min(h, w)//2)*2 + 1
        light = KF.gaussian_blur2d(light, (k, k), (25.0, 25.0))
        x = (x * light).clamp(0, 1)

    # vignette
    if random.random() < 0.55:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing="ij"
        )
        rr = torch.sqrt(xx**2 + yy**2)
        vig = (1.0 - torch.clamp((rr - 0.60) / 0.45, 0, 1) * random.uniform(0.05, 0.16)).unsqueeze(0).unsqueeze(0)
        k = 81 if min(h, w) >= 81 else (min(h, w)//2)*2 + 1
        vig = KF.gaussian_blur2d(vig, (k, k), (35.0, 35.0))
        x = (x * vig).clamp(0, 1)

    # blur
    if random.random() < 0.68:
        sigma = random.uniform(0.25, 1.1)
        k = 5 if sigma < 0.8 else 7
        x = KF.gaussian_blur2d(x, (k, k), (sigma, sigma)).clamp(0, 1)

    x = (x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# =========================================================
# docTR-shaped json
# =========================================================

def build_doctr_style_page_json(image_rel_path, img_h, img_w, page_items, page_idx=0):
    words = [w for w in page_items["words"] if (w["bbox"][2] - w["bbox"][0] >= 4 and w["bbox"][3] - w["bbox"][1] >= 4)]
    words.sort(key=lambda z: (z["block_id"], z["line_id"], z["bbox"][1], z["bbox"][0]))

    block_to_lines = defaultdict(lambda: defaultdict(list))
    for wd in words:
        block_to_lines[wd["block_id"]][wd["line_id"]].append(wd)

    blocks_json = []

    for block_id in sorted(block_to_lines.keys()):
        line_boxes = []
        lines_json = []

        for line_id in sorted(block_to_lines[block_id].keys()):
            line_words = sorted(block_to_lines[block_id][line_id], key=lambda z: z["bbox"][0])
            if not line_words:
                continue

            word_boxes = []
            word_entries = []

            for wd in line_words:
                x1, y1, x2, y2 = wd["bbox"]
                word_boxes.append([x1, y1, x2, y2])

                word_entries.append({
                    "value": wd["value"],
                    "org_text": wd["org_text"],
                    "confidence": 1.0,
                    "objectness_score": 1.0,
                    "geometry": normalize_box_xyxy(x1, y1, x2, y2, img_w, img_h),
                    "geometry_abs": [[int(x1), int(y1)], [int(x2), int(y2)]],
                    "quad": normalize_quad(wd["quad"], img_w, img_h),
                    "quad_abs": [[float(a), float(b)] for a, b in wd["quad"]],
                    "crop_orientation": {"value": 0, "confidence": 1.0}
                })

            lx1, ly1, lx2, ly2 = union_xyxy(word_boxes)
            line_boxes.append([lx1, ly1, lx2, ly2])

            lines_json.append({
                "geometry": normalize_box_xyxy(lx1, ly1, lx2, ly2, img_w, img_h),
                "geometry_abs": [[int(lx1), int(ly1)], [int(lx2), int(ly2)]],
                "objectness_score": 1.0,
                "confidence": 1.0,
                "words": word_entries
            })

        if not line_boxes:
            continue

        bx1, by1, bx2, by2 = union_xyxy(line_boxes)
        blocks_json.append({
            "geometry": normalize_box_xyxy(bx1, by1, bx2, by2, img_w, img_h),
            "geometry_abs": [[int(bx1), int(by1)], [int(bx2), int(by2)]],
            "objectness_score": 1.0,
            "confidence": 1.0,
            "lines": lines_json
        })

    return {
        "image_path": image_rel_path,
        "pages": [{
            "page_idx": page_idx,
            "dimensions": [img_h, img_w],
            "orientation": {"value": 0, "confidence": 1.0},
            "language": {"value": "kaithi", "confidence": 1.0},
            "blocks": blocks_json
        }]
    }


# =========================================================
# Worker init
# =========================================================

WORKER_STATE = {}


def init_worker(corpus_tokens, fonts, printed_font, bg_root, device_mode, base_seed):
    global WORKER_STATE
    pid = os.getpid()
    set_seed(base_seed + pid)

    WORKER_STATE["corpus_tokens"] = corpus_tokens
    WORKER_STATE["fonts"] = fonts
    WORKER_STATE["printed_font"] = printed_font
    WORKER_STATE["patch_db"] = load_patch_index(bg_root)
    WORKER_STATE["device"] = torch.device("cuda" if (device_mode == "cuda" and torch.cuda.is_available()) else "cpu")


# =========================================================
# Transliteration cache
# =========================================================

@lru_cache(maxsize=500000)
def translit_cached(text: str) -> str:
    return devanagari_to_kaithi(text)


def sample_dev_kaithi_pairs(n):
    corpus_tokens = WORKER_STATE["corpus_tokens"]
    chosen = random.choices(corpus_tokens, k=n)
    out = []
    for dev in chosen:
        kai = translit_cached(dev)
        if kai and kai.strip():
            out.append((dev, kai))
    return out


# =========================================================
# Page generation per worker
# =========================================================

def render_zone_words(page_rgba, zone, page_items):
    x1, y1, x2, y2 = zone["bbox"]
    zone_w = max(10, x2 - x1)
    zone_h = max(10, y2 - y1)
    ztype = zone["type"]
    n = zone["max_words"]

    if n == 0 or (ztype in ("remark", "remarks", "sig") and random.random() < 0.25):
        return

    if ztype in ("serial", "ref", "date") and random.random() < 0.60:
        pairs = []
        for _ in range(max(1, n)):
            if ztype == "date":
                dev = f"{random.randint(1,28)}-{random.randint(1,12)}-{random.randint(1,99)}"
            else:
                dev = f"{random.randint(1,99)}"
                if random.random() < 0.4:
                    dev += f"-{random.randint(1,99)}"
            kai = translit_cached(dev) or dev
            pairs.append((dev, kai))
    else:
        pairs = sample_dev_kaithi_pairs(max(1, n))

    if ztype == "sig" and random.random() < 0.45:
        pairs = [(random.choice(["/", "///", "—", "x", "✓"]), random.choice(["/", "///", "—", "x", "✓"]))]

    font_path = random.choice(WORKER_STATE["fonts"])
    font = fit_font_size_for_zone(pairs, zone_w, zone_h, font_path)

    temp = Image.new("RGBA", page_rgba.size, (255, 255, 255, 0))
    draw_tmp = ImageDraw.Draw(temp)
    _, _, _, sample_h = draw_tmp.textbbox((0, 0), "Ag", font=font)
    line_h = sample_h + random.randint(6, 12)

    cursor_x = x1 + random.randint(0, 6)
    cursor_y = y1 + random.randint(0, 4)

    base_angle = random.uniform(-2.0, 2.0) if ztype in ("paragraph", "footer_note", "margin_note", "particulars", "desc") else 0.0
    line_idx = 0
    current_line = []

    for dev_text, kai_text in pairs:
        bbox = draw_tmp.textbbox((0, 0), kai_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if cursor_x + tw > x2 - 4:
            if current_line:
                line_idx += 1
                current_line = []
            cursor_x = x1 + random.randint(0, 6)
            cursor_y += line_h

        if cursor_y + th > y2 - 2:
            break

        ann = draw_word_with_variation(
            temp,
            kai_text,
            cursor_x,
            cursor_y,
            font,
            angle_deg=base_angle + random.uniform(-1.2, 1.2)
        )
        if ann is not None:
            page_items["words"].append({
                "id": len(page_items["words"]),
                "block_id": zone["block_id"],
                "line_id": line_idx,
                "value": kai_text,
                "org_text": dev_text,
                "quad": ann["quad"],
                "bbox": ann["bbox"]
            })
            current_line.append(len(page_items["words"]) - 1)

        cursor_x += tw + random.randint(10, 22)

    page_rgba.alpha_composite(temp)


def create_clean_page(page_w, page_h):
    bg_cv = make_realistic_background(page_h, page_w, WORKER_STATE["patch_db"])
    page = cv_to_pil(bg_cv).convert("RGBA")
    draw = ImageDraw.Draw(page)

    printed_font = load_font(WORKER_STATE["printed_font"], random.randint(18, 24))

    if random.random() < 0.25:
        draw_page_border(draw, page_w, page_h)

    layout = pick_layout()
    if layout == "record":
        zones = draw_record_template(draw, page_w, page_h, printed_font)
    elif layout == "ledger":
        zones = draw_ledger_template(draw, page_w, page_h, printed_font)
    else:
        zones = draw_endorsement_template(draw, page_w, page_h, printed_font)

    if random.random() < 0.4:
        for _ in range(random.randint(20, 60)):
            x = random.randint(30, page_w - 30)
            y = random.randint(40, page_h - 40)
            draw.text((x, y), random.choice(["...", "—", ".", "·"]), font=printed_font, fill=(70, 70, 70))

    page_items = {"words": []}
    for z in zones:
        render_zone_words(page, z, page_items)

    if random.random() < 0.45:
        for _ in range(random.randint(1, 3)):
            sx = random.randint(int(page_w * 0.55), page_w - 120)
            sy = random.randint(int(page_h * 0.70), page_h - 60)
            sign_font = load_font(random.choice(WORKER_STATE["fonts"]), random.randint(18, 28))
            dev = random.choice(WORKER_STATE["corpus_tokens"])
            kai = translit_cached(dev)
            ann = draw_word_with_variation(page, kai, sx, sy, sign_font, angle_deg=random.uniform(-12, 12))
            if ann is not None:
                page_items["words"].append({
                    "id": len(page_items["words"]),
                    "block_id": max(z["block_id"] for z in zones) + 1,
                    "line_id": 0,
                    "value": kai,
                    "org_text": dev,
                    "quad": ann["quad"],
                    "bbox": ann["bbox"]
                })

    return page.convert("RGB"), page_items


def process_one_page(task):
    idx, out_dir, page_w, page_h = task
    device = WORKER_STATE["device"]

    page_style = random.choices(["A", "B", "C"], weights=[50, 35, 15], k=1)[0]
    page_pil, page_items = create_clean_page(page_w, page_h)

    img = pil_to_cv(page_pil)

    if random.random() < 0.78:
        img, page_items = perspective_transform_with_annotations(
            img, page_items, max_ratio=0.042 if page_style != "C" else 0.05
        )

    img = gpu_post_augment(img, device)

    if random.random() < (0.20 if page_style == "A" else 0.38 if page_style == "B" else 0.48):
        img = add_stains(img)
    if random.random() < (0.15 if page_style == "A" else 0.22 if page_style == "B" else 0.30):
        img = add_fold_lines(img)
    if random.random() < (0.10 if page_style == "A" else 0.18 if page_style == "B" else 0.25):
        img = add_bleed_through(img)
    if random.random() < (0.06 if page_style == "A" else 0.10 if page_style == "B" else 0.16):
        img = add_small_occlusion(img)

    img = jpeg_compress(img)

    page_name = f"page_{idx:06d}"
    img_path = os.path.join(out_dir, "images", page_name + ".jpg")
    ann_path = os.path.join(out_dir, "annotations", page_name + ".json")

    cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(75, 92)])

    h, w = img.shape[:2]
    ann = build_doctr_style_page_json(
        image_rel_path=os.path.relpath(img_path, out_dir),
        img_h=h,
        img_w=w,
        page_items=page_items,
        page_idx=0
    )
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)

    return {
        "idx": idx,
        "image_path": os.path.relpath(img_path, out_dir),
        "annotation_path": os.path.relpath(ann_path, out_dir)
    }


# =========================================================
# Split / PDF
# =========================================================

def assign_split(n, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    idxs = list(range(n))
    random.shuffle(idxs)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    split = {}

    for i in idxs[:n_train]:
        split[i] = "train"
    for i in idxs[n_train:n_train+n_val]:
        split[i] = "val"
    for i in idxs[n_train+n_val:]:
        split[i] = "test"
    return split


def copy_into_split_layout(out_dir, records, split_map):
    for split in ("train", "val", "test"):
        ensure_dir(os.path.join(out_dir, split, "images"))
        ensure_dir(os.path.join(out_dir, split, "annotations"))

    split_records = {"train": [], "val": [], "test": []}

    for rec in records:
        idx = rec["idx"]
        split = split_map[idx]

        src_img = os.path.join(out_dir, rec["image_path"])
        src_ann = os.path.join(out_dir, rec["annotation_path"])

        dst_img = os.path.join(out_dir, split, "images", os.path.basename(src_img))
        dst_ann = os.path.join(out_dir, split, "annotations", os.path.basename(src_ann))

        os.replace(src_img, dst_img)
        os.replace(src_ann, dst_ann)

        # rewrite annotation image_path
        with open(dst_ann, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["image_path"] = os.path.relpath(dst_img, out_dir)
        with open(dst_ann, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        split_records[split].append({
            "image": os.path.relpath(dst_img, out_dir),
            "annotation": os.path.relpath(dst_ann, out_dir),
            "split": split
        })

    return split_records


def build_pdf_from_images(image_paths, out_pdf):
    c = canvas.Canvas(out_pdf)
    for img_path in tqdm(image_paths, desc=f"Building {os.path.basename(out_pdf)}"):
        img = Image.open(img_path)
        w, h = img.size
        c.setPageSize((w, h))
        c.drawImage(ImageReader(img), 0, 0, width=w, height=h)
        c.showPage()
        img.close()
    c.save()


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--fonts_dir", required=True)
    ap.add_argument("--printed_font_path", required=True)
    ap.add_argument("--patch_root", required=True)
    ap.add_argument("--num_pages", type=int, default=1000)
    ap.add_argument("--page_w", type=int, default=1600)
    ap.add_argument("--page_h", type=int, default=2200)
    ap.add_argument("--train_ratio", type=float, default=0.90)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--make_pdfs", action="store_true")
    args = ap.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    set_seed(args.seed)
    print("making directories")
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "images"))
    ensure_dir(os.path.join(args.out_dir, "annotations"))
    ensure_dir(os.path.join(args.out_dir, "pdfs"))
    print("loading corpus tokens")
    corpus_tokens = load_devanagari_tokens(args.corpus_dir)
    print("loading font files")
    font_files = list_font_files(args.fonts_dir)
    if not font_files:
        raise ValueError(f"No fonts found in {args.fonts_dir}")

    fonts = [f for f in font_files if os.path.abspath(f) != os.path.abspath(args.printed_font_path)]
    if not fonts:
        fonts = font_files[:]
    print("creating tasks")
    tasks = [(i, args.out_dir, args.page_w, args.page_h) for i in range(args.num_pages)]
    records = []
    print("starting process pool")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(corpus_tokens, fonts, args.printed_font_path, args.patch_root, args.device, args.seed)
    ) as ex:
        futures = [ex.submit(process_one_page, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating pages"):
            records.append(fut.result())
    print("sorting records")
    records.sort(key=lambda x: x["idx"])
    print("assigning splits")
    split_map = assign_split(args.num_pages, args.train_ratio, args.val_ratio, args.test_ratio)
    print("copying into split layout")
    split_records = copy_into_split_layout(args.out_dir, records, split_map)

    for split in ("train", "val", "test"):
        with open(os.path.join(args.out_dir, f"{split}_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(split_records[split], f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            for rec in split_records[split]:
                f.write(rec["image"] + "\n")

    meta = {
        "num_pages": args.num_pages,
        "page_size": [args.page_w, args.page_h],
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "workers": args.workers,
        "device": args.device,
        "corpus_dir": args.corpus_dir,
        "fonts_dir": args.fonts_dir,
        "printed_font_path": args.printed_font_path,
        "patch_root": args.patch_root
    }
    with open(os.path.join(args.out_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.make_pdfs:
        for split in ("train", "val", "test"):
            image_paths = [os.path.join(args.out_dir, rec["image"]) for rec in split_records[split]]
            if image_paths:
                build_pdf_from_images(image_paths, os.path.join(args.out_dir, "pdfs", f"{split}.pdf"))

        all_paths = []
        for split in ("train", "val", "test"):
            all_paths.extend([os.path.join(args.out_dir, rec["image"]) for rec in split_records[split]])
        if all_paths:
            build_pdf_from_images(all_paths, os.path.join(args.out_dir, "pdfs", "all_pages.pdf"))

    print("\nDone.")
    print(f"Dataset saved to: {args.out_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()