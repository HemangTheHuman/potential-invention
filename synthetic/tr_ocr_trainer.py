import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
import regex as re
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)


# ============================================================
# 1. UTILS
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def grapheme_clusters(text: str) -> List[str]:
    return re.findall(r"\X", text)


def edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost
            )
    return int(dp[n, m])


def compute_cer(pred: str, gt: str, token_mode: str = "grapheme") -> float:
    if token_mode == "grapheme":
        p = grapheme_clusters(pred)
        g = grapheme_clusters(gt)
    else:
        p = list(pred)
        g = list(gt)

    if len(g) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return edit_distance(p, g) / len(g)


def compute_wer(pred: str, gt: str) -> float:
    p = pred.split()
    g = gt.split()
    if len(g) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return edit_distance(p, g) / len(g)


def safe_crop(image: Image.Image, box_abs: List[List[int]], pad: int = 2) -> Image.Image:
    (x1, y1), (x2, y2) = box_abs
    w, h = image.size

    x1 = max(0, int(math.floor(x1)) - pad)
    y1 = max(0, int(math.floor(y1)) - pad)
    x2 = min(w, int(math.ceil(x2)) + pad)
    y2 = min(h, int(math.ceil(y2)) + pad)

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return image.crop((x1, y1, x2, y2))


def paste_on_canvas(img: Image.Image, canvas_w: int, canvas_h: int, margin: int = 8) -> Image.Image:
    img = img.convert("RGB")
    usable_w = max(8, canvas_w - 2 * margin)
    usable_h = max(8, canvas_h - 2 * margin)

    scale = min(usable_w / img.width, usable_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))

    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    x = margin
    y = (canvas_h - new_h) // 2
    canvas.paste(img, (x, y))
    return canvas


# ============================================================
# 2. MANIFEST + ANNOTATION INDEXING
# ============================================================

def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]

    raise ValueError(f"Unsupported manifest format: {path}")


def build_word_samples(data_root: str, manifest_path: str, split_name: str) -> List[Dict[str, Any]]:
    root = Path(data_root)
    manifest = load_manifest(manifest_path)
    samples = []

    print(f"[index] split={split_name} manifest={manifest_path}")
    for item in tqdm(manifest, desc=f"Indexing-{split_name}"):
        image_rel = item["image"]
        ann_rel = item["annotation"]

        image_path = root / image_rel
        ann_path = root / ann_rel

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        pages = ann.get("pages", [])
        for page_idx, page in enumerate(pages):
            blocks = page.get("blocks", [])
            for block_idx, block in enumerate(blocks):
                lines = block.get("lines", [])
                for line_idx, line in enumerate(lines):
                    words = line.get("words", [])
                    for word_idx, word in enumerate(words):
                        text = (word.get("value") or "").strip()
                        box_abs = word.get("geometry_abs")

                        if not text:
                            continue
                        if not box_abs:
                            continue

                        samples.append({
                            "id": f"{split_name}_{Path(image_rel).stem}_p{page_idx}_b{block_idx}_l{line_idx}_w{word_idx}",
                            "split": split_name,
                            "image_path": str(image_path),
                            "annotation_path": str(ann_path),
                            "text": text,
                            "bbox_abs": box_abs,
                            "page_idx": page_idx,
                            "line_idx": line_idx,
                            "word_idx": word_idx,
                        })

    print(f"[index] split={split_name} total_word_samples={len(samples)}")
    return samples


# ============================================================
# 3. DATASET
# ============================================================

class KaithiWordTrOCRDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        processor: TrOCRProcessor,
        image_size: int = 384,
        crop_pad: int = 2,
        max_label_length: int = 64,
    ):
        self.samples = samples
        self.processor = processor
        self.image_size = image_size
        self.crop_pad = crop_pad
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]

        page = Image.open(row["image_path"]).convert("L")
        crop = safe_crop(page, row["bbox_abs"], pad=self.crop_pad)
        crop = paste_on_canvas(crop, self.image_size, self.image_size)

        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.squeeze(0)

        tokenized = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_label_length,
            return_tensors="pt",
        )

        labels = tokenized.input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": row["text"],
            "sample_id": row["id"],
            "image_path": row["image_path"],
            "bbox_abs": row["bbox_abs"],
        }


class TrOCRDataCollator:
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ============================================================
# 4. METRICS
# ============================================================

def build_compute_metrics(processor: TrOCRProcessor, token_mode: str = "grapheme"):
    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids.copy()

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        gt_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer_scores = []
        wer_scores = []
        exact = 0

        for pred, gt in zip(pred_texts, gt_texts):
            pred = pred.strip()
            gt = gt.strip()

            cer_scores.append(compute_cer(pred, gt, token_mode=token_mode))
            wer_scores.append(compute_wer(pred, gt))
            exact += int(pred == gt)

        metrics = {
            "cer": float(np.mean(cer_scores)) if cer_scores else 0.0,
            "wer": float(np.mean(wer_scores)) if wer_scores else 0.0,
            "exact_match": float(exact / len(gt_texts)) if gt_texts else 0.0,
        }
        return metrics

    return compute_metrics


# ============================================================
# 5. CALLBACKS
# ============================================================

class ConsoleLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        msg = f"[log] step={state.global_step}"
        for k, v in logs.items():
            if isinstance(v, float):
                msg += f" | {k}={v:.6f}"
            else:
                msg += f" | {k}={v}"
        print(msg)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"[epoch_end] epoch={state.epoch}")


class TensorBoardExtraCallback(TrainerCallback):
    def __init__(self, tb_logdir: str):
        self.writer = SummaryWriter(log_dir=tb_logdir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"hf/{k}", v, step)

        if torch.cuda.is_available():
            self.writer.add_scalar("system/gpu_mem_allocated_gb", torch.cuda.memory_allocated() / (1024**3), step)
            self.writer.add_scalar("system/gpu_mem_reserved_gb", torch.cuda.memory_reserved() / (1024**3), step)
            self.writer.add_scalar("system/gpu_max_mem_allocated_gb", torch.cuda.max_memory_allocated() / (1024**3), step)

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


class PredictionPreviewCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, raw_samples, tb_logdir: str, num_samples: int = 8):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.raw_samples = raw_samples
        self.num_samples = min(num_samples, len(eval_dataset))
        self.writer = SummaryWriter(log_dir=tb_logdir)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if self.num_samples == 0 or model is None:
            return

        model.eval()
        device = next(model.parameters()).device

        batch = [self.eval_dataset[i] for i in range(self.num_samples)]
        pixel_values = torch.stack([x["pixel_values"] for x in batch]).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=64,
                num_beams=1,
            )

        preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        lines = []
        for i in range(self.num_samples):
            gt = self.raw_samples[i]["text"]
            sid = self.raw_samples[i]["id"]
            pred = preds[i]
            lines.append(f"[{i}] {sid}\nGT   : {gt}\nPRED : {pred}\n")

        self.writer.add_text("eval/prediction_preview", "\n".join(lines), state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


# ============================================================
# 6. MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Kaithi word-level TrOCR fine-tuning")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, default="train_manifest.json")
    parser.add_argument("--val_manifest", type=str, default="val_manifest.json")
    parser.add_argument("--test_manifest", type=str, default="test_manifest.json")

    parser.add_argument("--model_name", type=str, default="microsoft/trocr-base-handwritten")
    parser.add_argument("--output_dir", type=str, default="./trocr_kaithi_word_ckpts")
    parser.add_argument("--tb_logdir", type=str, default="./runs/trocr_kaithi_word")

    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--crop_pad", type=int, default=2)
    parser.add_argument("--max_label_length", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=12)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--token_mode", type=str, default="grapheme", choices=["char", "grapheme"])

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tb_logdir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training pipeline.")

    if args.bf16 and args.fp16:
        raise ValueError("Use only one of --bf16 or --fp16")

    print("=" * 80)
    print("Kaithi TrOCR word-level fine-tuning")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("=" * 80)
    print(f"[device] {torch.cuda.get_device_name(0)}")
    print(f"[cuda] {torch.version.cuda}")

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    # generation settings must go here, not in model.config
    model.generation_config.max_length = args.max_label_length
    model.generation_config.num_beams = args.num_beams
    model.generation_config.early_stopping = True
    model.generation_config.length_penalty = 1.0
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    # model.config.max_length = args.max_label_length
    # model.config.num_beams = args.num_beams
    # model.config.early_stopping = True
    # model.config.length_penalty = 1.0
    # model.config.no_repeat_ngram_size = 0

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_manifest = str(Path(args.data_root) / args.train_manifest)
    val_manifest = str(Path(args.data_root) / args.val_manifest)
    test_manifest = str(Path(args.data_root) / args.test_manifest)

    train_samples = build_word_samples(args.data_root, train_manifest, "train")
    val_samples = build_word_samples(args.data_root, val_manifest, "val")
    test_samples = build_word_samples(args.data_root, test_manifest, "test")

    print(f"[dataset] train_word_samples={len(train_samples)}")
    print(f"[dataset] val_word_samples={len(val_samples)}")
    print(f"[dataset] test_word_samples={len(test_samples)}")

    train_ds = KaithiWordTrOCRDataset(
        train_samples,
        processor=processor,
        image_size=args.image_size,
        crop_pad=args.crop_pad,
        max_label_length=args.max_label_length,
    )
    val_ds = KaithiWordTrOCRDataset(
        val_samples,
        processor=processor,
        image_size=args.image_size,
        crop_pad=args.crop_pad,
        max_label_length=args.max_label_length,
    )
    test_ds = KaithiWordTrOCRDataset(
        test_samples,
        processor=processor,
        image_size=args.image_size,
        crop_pad=args.crop_pad,
        max_label_length=args.max_label_length,
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_ds) / (args.train_batch_size * args.gradient_accumulation_steps)
    )
    total_train_steps = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    os.environ["TENSORBOARD_LOGGING_DIR"] = args.tb_logdir
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=not args.eval_only,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        fp16=args.fp16,
        bf16=args.bf16,
        predict_with_generate=True,
        generation_max_length=args.max_label_length,
        generation_num_beams=args.num_beams,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=["labels"],
    
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=TrOCRDataCollator(),
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor, token_mode=args.token_mode),
        callbacks=[
            ConsoleLoggerCallback(),
            TensorBoardExtraCallback(args.tb_logdir),
            PredictionPreviewCallback(processor, val_ds, val_samples, args.tb_logdir, num_samples=8),
        ],
    )

    if not args.eval_only:
        print("[train] Starting training...")
        train_result = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint else None
        )
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        print("[train] Training complete.")

    print("[eval] Running validation...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    trainer.log_metrics("val", val_metrics)
    trainer.save_metrics("val", val_metrics)
    print(json.dumps(val_metrics, indent=2))

    print("[eval] Running test...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    print(json.dumps(test_metrics, indent=2))

    with open(Path(args.output_dir) / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2, ensure_ascii=False)

    print(f"[done] saved final metrics to {Path(args.output_dir) / 'final_metrics.json'}")


if __name__ == "__main__":
    main()