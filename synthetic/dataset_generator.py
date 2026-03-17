import os
import cv2
import json
import math
import random
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple
from aksharamukha import transliterate
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from tqdm import tqdm
import regex as re


# ============================================================
# 1. CONFIG
# ============================================================

CONFIG = {
    "seed": 42,
    "output_dir": "output",
    "fonts_dir": "fonts",
    "source_text_path": "corpus/english_source.txt",

    # dataset sizing
    "num_word_samples": 100000,
    "num_line_samples": 20000,

    # difficulty ratio
    "difficulty_probs": {
        "clean": 0.40,
        "mild": 0.40,
        "hard": 0.20,
    },

    # word/line settings
    "line_min_words": 3,
    "line_max_words": 10,
    "min_word_len": 1,
    "max_word_len": 18,

    # image sizes
    "word_canvas_h": 96,
    "line_canvas_h": 128,
    "max_canvas_w": 1200,

    # font settings
    "font_size_word": (36, 60),
    "font_size_line": (28, 52),

    # margins
    "margin_x": (12, 40),
    "margin_y": (8, 26),

    # split ratio
    "split_probs": {
        "train": 0.90,
        "val": 0.08,
        "test": 0.02,
    },

    # rendering diversity
    "bg_color_range": (235, 255),
    "fg_color_range": (0, 40),
    "tracking_range": (-1, 3),  # per-char spacing tweak

    # background texture probability
    "background_texture_prob": 0.35,

    # save format
    "image_ext": ".png",
}


# ============================================================
# 2. USER-SUPPLIED TRANSLITERATION FUNCTION
# Replace this with your real transliterator.
# ============================================================

def transliterate_english_to_kaithi(text: str) -> str:
    """
    Replace this with your package call.

    Example:
        from your_package import transliterate
        return transliterate(text)

    This placeholder returns input unchanged.
    """
    src='HK'
    tgt='Kaithi'
    txt=text.lower()
    res=transliterate.process(src, tgt, txt, nativize =False, pre_options = [], post_options = [])
    return res


# ============================================================
# 3. TEXT NORMALIZATION + TOKENIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Consistent Unicode normalization.
    NFC is usually safer for OCR label consistency than leaving raw.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFC", text)
    return text


def grapheme_clusters(text: str) -> List[str]:
    """
    Uses regex \\X to split into Unicode grapheme clusters.
    """
    return re.findall(r"\X", text)


def chars(text: str) -> List[str]:
    return list(text)


# ============================================================
# 4. LOAD CORPUS
# ============================================================

def load_source_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def build_word_pool(lines: List[str]) -> List[str]:
    words = []
    for line in lines:
        parts = re.split(r"\s+", line.strip())
        for p in parts:
            clean = p.strip()
            if clean:
                words.append(clean)
    return words


# ============================================================
# 5. HELPERS
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def choose_split() -> str:
    r = random.random()
    train_p = CONFIG["split_probs"]["train"]
    val_p = CONFIG["split_probs"]["val"]
    if r < train_p:
        return "train"
    if r < train_p + val_p:
        return "val"
    return "test"


def choose_difficulty() -> str:
    probs = CONFIG["difficulty_probs"]
    r = random.random()
    acc = 0.0
    for k, v in probs.items():
        acc += v
        if r <= acc:
            return k
    return "hard"


def stable_uid(text: str, sample_type: str, idx: int) -> str:
    h = hashlib.md5(f"{sample_type}|{idx}|{text}".encode("utf-8")).hexdigest()[:12]
    return h


def ensure_dirs(base_dir: str):
    for split in ["train", "val", "test"]:
        for sample_type in ["word", "line"]:
            for diff in ["clean", "mild", "hard"]:
                os.makedirs(os.path.join(base_dir, split, sample_type, diff), exist_ok=True)


def load_fonts(fonts_dir: str) -> List[str]:
    font_paths = []
    for p in Path(fonts_dir).glob("*.ttf"):
        font_paths.append(str(p))
    for p in Path(fonts_dir).glob("*.otf"):
        font_paths.append(str(p))
    if not font_paths:
        raise FileNotFoundError(f"No .ttf/.otf fonts found in {fonts_dir}")
    return font_paths


def random_bg_color() -> int:
    lo, hi = CONFIG["bg_color_range"]
    return random.randint(lo, hi)


def random_fg_color() -> int:
    lo, hi = CONFIG["fg_color_range"]
    return random.randint(lo, hi)


# ============================================================
# 6. TEXT SAMPLING
# ============================================================

def sample_word(word_pool: List[str]) -> str:
    while True:
        w = random.choice(word_pool).strip()
        if CONFIG["min_word_len"] <= len(w) <= CONFIG["max_word_len"]:
            return w


def sample_line(word_pool: List[str]) -> str:
    n = random.randint(CONFIG["line_min_words"], CONFIG["line_max_words"])
    return " ".join(sample_word(word_pool) for _ in range(n))


# ============================================================
# 7. RENDERING
# ============================================================

def get_text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def draw_text_with_tracking(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: int,
    tracking: int = 0,
    ):
    x, y = xy
    for cluster in grapheme_clusters(text):
        draw.text((x, y), cluster, font=font, fill=fill)
        bbox = draw.textbbox((x, y), cluster, font=font)
        cluster_w = bbox[2] - bbox[0]
        x += cluster_w + tracking


def render_text_image(
    text: str,
    sample_type: str,
    font_path: str,
    ) -> Image.Image:
    """
    Render text to grayscale PIL image.
    """
    if sample_type == "word":
        font_size = random.randint(*CONFIG["font_size_word"])
        canvas_h = CONFIG["word_canvas_h"]
    else:
        font_size = random.randint(*CONFIG["font_size_line"])
        canvas_h = CONFIG["line_canvas_h"]

    font = ImageFont.truetype(font_path, font_size)

    temp_img = Image.new("L", (CONFIG["max_canvas_w"], canvas_h), color=random_bg_color())
    temp_draw = ImageDraw.Draw(temp_img)

    margin_x = random.randint(*CONFIG["margin_x"])
    margin_y = random.randint(*CONFIG["margin_y"])
    tracking = random.randint(*CONFIG["tracking_range"])
    fg = random_fg_color()

    # Estimate size with plain bbox first
    bbox = get_text_bbox(temp_draw, text, font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    canvas_w = min(max(text_w + 2 * margin_x + 40, 120), CONFIG["max_canvas_w"])
    img = Image.new("L", (canvas_w, canvas_h), color=random_bg_color())
    draw = ImageDraw.Draw(img)

    # vertically center
    y = max(0, (canvas_h - text_h) // 2 - bbox[1] + random.randint(-3, 3))
    x = margin_x

    draw_text_with_tracking(draw, (x, y), text, font, fill=fg, tracking=tracking)

    # Optional underline / stray mark very rarely
    if random.random() < 0.03:
        yline = min(canvas_h - 2, y + text_h + random.randint(2, 6))
        draw.line((x, yline, min(canvas_w - 10, x + text_w), yline), fill=random.randint(80, 140), width=1)

    img = trim_whitespace(img, pad=random.randint(4, 16))
    return img


def trim_whitespace(img: Image.Image, pad: int = 8) -> Image.Image:
    arr = np.array(img)
    mask = arr < 250
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(arr.shape[1], x1 + pad)
    y1 = min(arr.shape[0], y1 + pad)

    cropped = arr[y0:y1, x0:x1]
    return Image.fromarray(cropped)


# ============================================================
# 8. AUGMENTATIONS
# clean / mild / hard
# ============================================================

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def add_paper_texture(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    noise = np.random.normal(loc=0, scale=random.uniform(3, 10), size=(h, w))
    gradient = np.tile(np.linspace(random.uniform(-8, 8), random.uniform(-8, 8), w), (h, 1))
    textured = arr.astype(np.float32) + noise + gradient
    return np.clip(textured, 0, 255).astype(np.uint8)


def apply_gaussian_blur(arr: np.ndarray, max_ksize: int = 5) -> np.ndarray:
    k = random.choice([3, 5]) if max_ksize >= 5 else 3
    return cv2.GaussianBlur(arr, (k, k), random.uniform(0.2, 1.3))


def apply_motion_blur(arr: np.ndarray, ksize: int = 7) -> np.ndarray:
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    if random.random() < 0.5:
        kernel[ksize // 2, :] = 1.0
    else:
        kernel[:, ksize // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(arr, -1, kernel)


def jpeg_compress(arr: np.ndarray, quality_range=(25, 70)) -> np.ndarray:
    quality = random.randint(*quality_range)
    success, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return arr
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return dec


def low_dpi_simulation(arr: np.ndarray, min_scale=0.35, max_scale=0.75) -> np.ndarray:
    h, w = arr.shape
    scale = random.uniform(min_scale, max_scale)
    small = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def random_affine(arr: np.ndarray, max_rot=3.0, max_shear=0.04, max_shift=0.04) -> np.ndarray:
    h, w = arr.shape
    rot = random.uniform(-max_rot, max_rot)
    shear = random.uniform(-max_shear, max_shear)
    tx = random.uniform(-max_shift, max_shift) * w
    ty = random.uniform(-max_shift, max_shift) * h

    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
    M[0, 1] += shear
    M[0, 2] += tx
    M[1, 2] += ty

    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def elastic_distortion(arr: np.ndarray, alpha=8.0, sigma=5.0) -> np.ndarray:
    """
    Light elastic deformation.
    """
    h, w = arr.shape
    dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_gaussian_noise(arr: np.ndarray, sigma_range=(4, 18)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, arr.shape)
    out = arr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(arr: np.ndarray, amount=0.003) -> np.ndarray:
    out = arr.copy()
    h, w = arr.shape
    num = int(amount * h * w)

    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    out[ys, xs] = np.random.choice([0, 255], size=num)
    return out


def ink_erode(arr: np.ndarray) -> np.ndarray:
    k = random.choice([1, 2])
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(arr, kernel, iterations=1)


def ink_dilate(arr: np.ndarray) -> np.ndarray:
    k = random.choice([1, 2])
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(arr, kernel, iterations=1)


def random_contrast(arr: np.ndarray, alpha_range=(0.85, 1.2), beta_range=(-12, 12)) -> np.ndarray:
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    out = arr.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def vignetting(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    mask = 1 - np.clip((dist - 0.2), 0, 1) * random.uniform(10, 35)
    out = arr.astype(np.float32) + mask
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_clean(img: Image.Image) -> Image.Image:
    arr = pil_to_np(img)

    if random.random() < CONFIG["background_texture_prob"] * 0.2:
        arr = add_paper_texture(arr)

    if random.random() < 0.15:
        arr = random_affine(arr, max_rot=1.0, max_shear=0.01, max_shift=0.01)

    arr = random_contrast(arr, alpha_range=(0.95, 1.05), beta_range=(-4, 4))
    return np_to_pil(arr)


def augment_mild(img: Image.Image) -> Image.Image:
    arr = pil_to_np(img)

    if random.random() < 0.5:
        arr = add_paper_texture(arr)

    if random.random() < 0.6:
        arr = random_affine(arr, max_rot=2.0, max_shear=0.02, max_shift=0.02)

    if random.random() < 0.55:
        arr = apply_gaussian_blur(arr, max_ksize=5)

    if random.random() < 0.45:
        arr = low_dpi_simulation(arr, min_scale=0.55, max_scale=0.85)

    if random.random() < 0.50:
        arr = jpeg_compress(arr, quality_range=(45, 80))

    if random.random() < 0.45:
        arr = add_gaussian_noise(arr, sigma_range=(3, 10))

    if random.random() < 0.25:
        arr = add_salt_pepper(arr, amount=0.0015)

    arr = random_contrast(arr, alpha_range=(0.88, 1.12), beta_range=(-10, 10))
    return np_to_pil(arr)


def augment_hard(img: Image.Image) -> Image.Image:
    arr = pil_to_np(img)

    if random.random() < 0.8:
        arr = add_paper_texture(arr)

    if random.random() < 0.7:
        arr = random_affine(arr, max_rot=4.0, max_shear=0.04, max_shift=0.04)

    if random.random() < 0.55:
        arr = elastic_distortion(arr, alpha=random.uniform(4, 10), sigma=random.uniform(4, 6))

    if random.random() < 0.6:
        arr = apply_gaussian_blur(arr, max_ksize=5)

    if random.random() < 0.5:
        arr = apply_motion_blur(arr, ksize=random.choice([5, 7, 9]))

    if random.random() < 0.7:
        arr = low_dpi_simulation(arr, min_scale=0.35, max_scale=0.7)

    if random.random() < 0.7:
        arr = jpeg_compress(arr, quality_range=(20, 65))

    if random.random() < 0.7:
        arr = add_gaussian_noise(arr, sigma_range=(6, 18))

    if random.random() < 0.4:
        arr = add_salt_pepper(arr, amount=0.003)

    if random.random() < 0.4:
        arr = ink_erode(arr)

    if random.random() < 0.4:
        arr = ink_dilate(arr)

    arr = random_contrast(arr, alpha_range=(0.78, 1.18), beta_range=(-16, 16))
    return np_to_pil(arr)


def augment_by_difficulty(img: Image.Image, difficulty: str) -> Image.Image:
    if difficulty == "clean":
        return augment_clean(img)
    elif difficulty == "mild":
        return augment_mild(img)
    return augment_hard(img)


# ============================================================
# 9. SAVE SAMPLE + MANIFEST
# ============================================================

def save_sample(
    img: Image.Image,
    out_path: str,
):
    img.save(out_path)


def manifest_record(
    uid: str,
    rel_path: str,
    sample_type: str,
    difficulty: str,
    split: str,
    text: str,
    src_text: str,
    font_path: str,
    ) -> Dict:
    return {
        "id": uid,
        "image_path": rel_path.replace("\\", "/"),
        "sample_type": sample_type,
        "difficulty": difficulty,
        "split": split,
        "text": text,
        "src_text": src_text,
        "char_tokens": chars(text),
        "grapheme_tokens": grapheme_clusters(text),
        "num_chars": len(chars(text)),
        "num_graphemes": len(grapheme_clusters(text)),
        "font": os.path.basename(font_path),
    }


# ============================================================
# 10. DATASET GENERATION
# ============================================================

def generate_dataset():
    seed_everything(CONFIG["seed"])

    out_dir = CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ensure_dirs(out_dir)

    font_paths = load_fonts(CONFIG["fonts_dir"])
    source_lines = load_source_lines(CONFIG["source_text_path"])
    word_pool = build_word_pool(source_lines)

    manifests = {
        "train": [],
        "val": [],
        "test": [],
    }

    # -------- WORD SAMPLES --------
    for i in tqdm(range(CONFIG["num_word_samples"]), desc="Generating word samples"):
        split = choose_split()
        difficulty = choose_difficulty()
        source_text = sample_word(word_pool)
        kaithi_text = normalize_text(transliterate_english_to_kaithi(source_text))

        if not kaithi_text:
            continue

        font_path = random.choice(font_paths)
        img = render_text_image(kaithi_text, sample_type="word", font_path=font_path)
        img = augment_by_difficulty(img, difficulty)

        uid = stable_uid(kaithi_text, "word", i)
        filename = f"{uid}{CONFIG['image_ext']}"
        rel_path = os.path.join(split, "word", difficulty, filename)
        abs_path = os.path.join(out_dir, rel_path)

        save_sample(img, abs_path)
        manifests[split].append(
            manifest_record(
                uid=uid,
                rel_path=rel_path,
                sample_type="word",
                difficulty=difficulty,
                split=split,
                text=kaithi_text,
                font_path=font_path,
                src_text=source_text,
            )
        )

    # -------- LINE SAMPLES --------
    for i in tqdm(range(CONFIG["num_line_samples"]), desc="Generating line samples"):
        split = choose_split()
        difficulty = choose_difficulty()
        source_text = sample_line(word_pool)
        kaithi_text = normalize_text(transliterate_english_to_kaithi(source_text))

        if not kaithi_text:
            continue

        font_path = random.choice(font_paths)
        img = render_text_image(kaithi_text, sample_type="line", font_path=font_path)
        img = augment_by_difficulty(img, difficulty)

        uid = stable_uid(kaithi_text, "line", i)
        filename = f"{uid}{CONFIG['image_ext']}"
        rel_path = os.path.join(split, "line", difficulty, filename)
        abs_path = os.path.join(out_dir, rel_path)

        save_sample(img, abs_path)
        manifests[split].append(
            manifest_record(
                uid=uid,
                rel_path=rel_path,
                sample_type="line",
                difficulty=difficulty,
                split=split,
                text=kaithi_text,
                font_path=font_path,
                src_text=source_text,
            )
        )

    # save JSONL manifests
    for split, records in manifests.items():
        manifest_path = os.path.join(out_dir, f"{split}.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # save vocab files
    save_vocab_files(manifests, out_dir)

    print("Done.")
    print(f"Saved dataset to: {out_dir}")


def save_vocab_files(manifests: Dict[str, List[Dict]], out_dir: str):
    all_text = []
    for split_records in manifests.values():
        for rec in split_records:
            all_text.append(rec["text"])

    joined = "\n".join(all_text)
    char_vocab = sorted(set(chars(joined)))
    grapheme_vocab = sorted(set(grapheme_clusters(joined)))

    with open(os.path.join(out_dir, "char_vocab.txt"), "w", encoding="utf-8") as f:
        for token in char_vocab:
            f.write(token + "\n")

    with open(os.path.join(out_dir, "grapheme_vocab.txt"), "w", encoding="utf-8") as f:
        for token in grapheme_vocab:
            f.write(token + "\n")


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_dataset()