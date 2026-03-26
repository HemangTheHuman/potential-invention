import os
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader

from RCNN_trainer import CRNN, CTCTokenizer, greedy_decode


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def polygon_to_aabb(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "xmin": int(math.floor(min(xs))),
        "ymin": int(math.floor(min(ys))),
        "xmax": int(math.ceil(max(xs))),
        "ymax": int(math.ceil(max(ys))),
    }


def normalize_bbox(xmin, ymin, xmax, ymax, img_w, img_h, pad=2):
    xmin = clamp(int(math.floor(xmin)) - pad, 0, img_w - 1)
    ymin = clamp(int(math.floor(ymin)) - pad, 0, img_h - 1)
    xmax = clamp(int(math.ceil(xmax)) + pad, 1, img_w)
    ymax = clamp(int(math.ceil(ymax)) + pad, 1, img_h)

    if xmax <= xmin:
        xmax = min(img_w, xmin + 1)
    if ymax <= ymin:
        ymax = min(img_h, ymin + 1)

    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def word_to_bbox(word, img_w, img_h, pad=2):
    if "quad_abs" in word and word["quad_abs"]:
        bbox = polygon_to_aabb(word["quad_abs"])
        return normalize_bbox(
            bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"], img_w, img_h, pad=pad
        )

    if "geometry_abs" in word and word["geometry_abs"]:
        (x1, y1), (x2, y2) = word["geometry_abs"]
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return normalize_bbox(xmin, ymin, xmax, ymax, img_w, img_h, pad=pad)

    return None


def extract_words_from_nested_json(data, page_img):
    img_w, img_h = page_img.size
    words_meta = []

    pages = data.get("pages", [])
    for page_idx, page in enumerate(pages):
        blocks = page.get("blocks", [])
        for block_idx, block in enumerate(blocks):
            lines = block.get("lines", [])
            for line_idx, line in enumerate(lines):
                words = line.get("words", [])
                for word_idx, word in enumerate(words):
                    bbox = word_to_bbox(word, img_w, img_h, pad=2)
                    if bbox is None:
                        continue

                    words_meta.append(
                        {
                            "page_idx": page_idx,
                            "block_idx": block_idx,
                            "line_idx": line_idx,
                            "word_idx": word_idx,
                            "bbox": bbox,
                            "word_ref": word,
                            "original_value": word.get("value", ""),
                            "org_text": word.get("org_text", ""),
                        }
                    )
    return words_meta


class JSONWordCropDataset(Dataset):
    def __init__(self, page_image, words_meta, target_h=64, max_width=768):
        self.page_image = page_image.convert("L")
        self.words_meta = words_meta
        self.target_h = target_h
        self.max_width = max_width

    def __len__(self):
        return len(self.words_meta)

    def __getitem__(self, idx):
        meta = self.words_meta[idx]
        bbox = meta["bbox"]
        xmin, ymin, xmax, ymax = (
            bbox["xmin"],
            bbox["ymin"],
            bbox["xmax"],
            bbox["ymax"],
        )

        cropped = self.page_image.crop((xmin, ymin, xmax, ymax))
        arr = np.array(cropped)

        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            x = torch.zeros((1, self.target_h, 32), dtype=torch.float32)
            return x, idx

        scale = self.target_h / h
        new_w = max(32, int(w * scale))
        new_w = min(new_w, self.max_width)

        resized = Image.fromarray(arr).resize((new_w, self.target_h), Image.BILINEAR)
        arr_resized = np.array(resized)

        x = torch.from_numpy(arr_resized).float().unsqueeze(0) / 255.0
        return x, idx


def collate_inference(batch):
    tensors = [item[0] for item in batch]
    indices = [item[1] for item in batch]

    widths = [t.shape[-1] for t in tensors]
    max_w = max(widths)
    B = len(tensors)
    H = tensors[0].shape[-2]

    x_batch = torch.ones((B, 1, H, max_w), dtype=torch.float32)
    for j, t in enumerate(tensors):
        w = t.shape[-1]
        x_batch[j, :, :, :w] = t

    return x_batch, indices


def load_model_and_tokenizer(ckpt_path, vocab_path, device):
    tokenizer = CTCTokenizer(vocab_path)
    model = CRNN(
        num_classes=tokenizer.vocab_size,
        lstm_hidden=384,
        lstm_layers=3,
        dropout=0.2,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    if list(state_dict.keys())[0].startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, tokenizer


def draw_text_box(draw, x, y, text, font):
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        tx1, ty1, tx2, ty2 = bbox
    except Exception:
        w, h = draw.textsize(text, font=font)
        tx1, ty1, tx2, ty2 = x, y, x + w, y + h

    draw.rectangle([tx1 - 2, ty1 - 1, tx2 + 2, ty2 + 1], fill="white", outline="red", width=1)
    draw.text((x, y), text, fill="red", font=font)


def annotate_full_page(page_image, words_meta, output_path):
    page_rgb = page_image.convert("RGB")
    draw = ImageDraw.Draw(page_rgb)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for meta in words_meta:
        bbox = meta["bbox"]
        pred = meta.get("predicted_value", "")
        xmin, ymin, xmax, ymax = (
            bbox["xmin"],
            bbox["ymin"],
            bbox["xmax"],
            bbox["ymax"],
        )

        draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=2)

        label = pred if pred else "?"
        text_y = max(0, ymin - 14)
        draw_text_box(draw, xmin, text_y, label, font)

    page_rgb.save(output_path)


def build_output_json_structure(input_data, words_meta):
    output = {
        "image_path": input_data.get("image_path", ""),
        "pages": []
    }

    page_map = {}

    for meta in words_meta:
        page_idx = meta["page_idx"]
        if page_idx not in page_map:
            page_map[page_idx] = []

        page_map[page_idx].append(
            {
                "block_idx": meta["block_idx"],
                "line_idx": meta["line_idx"],
                "word_idx": meta["word_idx"],
                "bbox": meta["bbox"],
                "original_value": meta.get("original_value", ""),
                "org_text": meta.get("org_text", ""),
                "predicted_value": meta.get("predicted_value", ""),
            }
        )

    for page_idx in sorted(page_map.keys()):
        output["pages"].append(
            {
                "page_idx": page_idx,
                "words": page_map[page_idx]
            }
        )

    return output


def process_image_and_json(
    image_path,
    json_path,
    output_dir,
    model,
    tokenizer,
    device,
    amp_dtype,
    batch_size=256,
    num_workers=8,
    max_width=768,
    target_h=64,
):
    ensure_dir(output_dir)

    print(f"Loading image: {image_path}")
    page_img = Image.open(image_path).convert("RGB")

    print(f"Loading JSON: {json_path}")
    data = load_json(json_path)

    words_meta = extract_words_from_nested_json(data, page_img)
    print(f"Found {len(words_meta)} word boxes")

    if len(words_meta) == 0:
        print("No word boxes found in JSON.")
        return

    dataset = JSONWordCropDataset(
        page_image=page_img,
        words_meta=words_meta,
        target_h=target_h,
        max_width=max_width,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_inference,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    start_time = time.time()

    all_preds = [""] * len(words_meta)

    for batch, indices in loader:
        batch = batch.to(device, non_blocking=True)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            logits = model(batch)
            preds = greedy_decode(logits, tokenizer)

        for ds_idx, pred in zip(indices, preds):
            all_preds[ds_idx] = pred

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    print(f"Inference done in {elapsed:.2f}s")

    for i, meta in enumerate(words_meta):
        pred = all_preds[i]
        meta["predicted_value"] = pred

    annotated_page_path = os.path.join(output_dir, "annotated_page.png")
    annotate_full_page(page_img, words_meta, annotated_page_path)

    output_json = build_output_json_structure(data, words_meta)
    output_json_path = os.path.join(output_dir, "predictions.json")
    save_json(output_json, output_json_path)

    print("-" * 60)
    print(f"Annotated page saved to: {annotated_page_path}")
    print(f"Predictions JSON saved to: {output_json_path}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained CRNN checkpoint")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab txt used during training")
    parser.add_argument("--image", type=str, required=True, help="Path to the page image")
    parser.add_argument("--json", type=str, required=True, help="Path to OCR JSON with nested pages->blocks->lines->words")
    parser.add_argument("--output_dir", type=str, default="./ocr_inference_output", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_width", type=int, default=768)
    parser.add_argument("--target_h", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model, tokenizer = load_model_and_tokenizer(args.ckpt, args.vocab, device)
    print("Model loaded successfully.")

    process_image_and_json(
        image_path=args.image,
        json_path=args.json,
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        device=device,
        amp_dtype=amp_dtype,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_width=args.max_width,
        target_h=args.target_h,
    )


if __name__ == "__main__":
    main()