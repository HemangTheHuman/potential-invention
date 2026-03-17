import os
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import regex as re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# 1. UTILS
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def compute_cer(pred: str, gt: str, token_mode: str = "char") -> float:
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


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_examples_for_tensorboard(preds: List[str], gts: List[str], image_paths: List[str], max_examples: int = 8) -> str:
    lines = []
    k = min(max_examples, len(preds))
    for i in range(k):
        lines.append(
            f"[{i}] path={image_paths[i]}\nGT   : {gts[i]}\nPRED : {preds[i]}\n"
        )
    return "\n".join(lines)


# ============================================================
# 2. TOKENIZER
# ============================================================

class CTCTokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokens = [line.rstrip("\n") for line in f]

        self.blank_id = 0
        self.pad_id = 0
        self.id2tok = ["<BLANK>"] + tokens
        self.tok2id = {tok: i for i, tok in enumerate(self.id2tok)}
        self.vocab_size = len(self.id2tok)

    def encode(self, tokens: List[str]) -> List[int]:
        ids = []
        for t in tokens:
            if t in self.tok2id:
                ids.append(self.tok2id[t])
        return ids

    def decode_ctc(self, ids: List[int]) -> List[str]:
        out = []
        prev = None
        for idx in ids:
            if idx == self.blank_id:
                prev = idx
                continue
            if idx != prev:
                out.append(self.id2tok[idx])
            prev = idx
        return out

    def decode_to_text(self, ids: List[int]) -> str:
        return "".join(self.decode_ctc(ids))


# ============================================================
# 3. DATASET
# ============================================================

@dataclass
class DatasetConfig:
    root_dir: str
    jsonl_path: str
    token_mode: str = "grapheme"
    sample_type: str = "word"
    min_width: int = 32
    img_height: int = 64
    max_width: int = 1024
    invert: bool = False
    difficulty_filter: Optional[List[str]] = None


class KaithiOCRDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, tokenizer: CTCTokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        rows = load_jsonl(cfg.jsonl_path)

        filtered = []
        for r in rows:
            if cfg.sample_type != "both" and r["sample_type"] != cfg.sample_type:
                continue
            if cfg.difficulty_filter is not None and r["difficulty"] not in cfg.difficulty_filter:
                continue
            filtered.append(r)

        self.rows = filtered
        self.root_dir = Path(cfg.root_dir)

    def __len__(self):
        return len(self.rows)

    def _load_image(self, rel_path: str) -> np.ndarray:
        img = Image.open(self.root_dir / rel_path).convert("L")
        arr = np.array(img)
        if self.cfg.invert:
            arr = 255 - arr
        return arr

    def _resize_keep_aspect(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        target_h = self.cfg.img_height
        scale = target_h / h
        new_w = max(self.cfg.min_width, int(w * scale))
        new_w = min(new_w, self.cfg.max_width)
        resized = Image.fromarray(arr).resize((new_w, target_h), Image.BILINEAR)
        return np.array(resized)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        arr = self._load_image(row["image_path"])
        arr = self._resize_keep_aspect(arr)

        x = torch.from_numpy(arr).float().unsqueeze(0) / 255.0

        label_tokens = row["grapheme_tokens"] if self.cfg.token_mode == "grapheme" else row["char_tokens"]
        y = self.tokenizer.encode(label_tokens)

        return {
            "image": x,
            "label_ids": torch.tensor(y, dtype=torch.long),
            "text": row["text"],
            "difficulty": row["difficulty"],
            "sample_type": row["sample_type"],
            "image_path": row["image_path"],
        }


# ============================================================
# 4. COLLATE
# ============================================================

def collate_fn(batch: List[Dict]):
    batch = sorted(batch, key=lambda x: x["image"].shape[-1], reverse=True)

    images = [b["image"] for b in batch]
    widths = [img.shape[-1] for img in images]
    heights = [img.shape[-2] for img in images]

    max_w = max(widths)
    B = len(images)
    H = heights[0]

    x = torch.ones((B, 1, H, max_w), dtype=torch.float32)
    for i, img in enumerate(images):
        w = img.shape[-1]
        x[i, :, :, :w] = img

    label_ids = [b["label_ids"] for b in batch]
    target_lengths = torch.tensor([len(y) for y in label_ids], dtype=torch.long)
    labels_concat = torch.cat(label_ids, dim=0) if len(label_ids) > 0 else torch.empty(0, dtype=torch.long)

    return {
        "images": x,
        "labels_concat": labels_concat,
        "target_lengths": target_lengths,
        "texts": [b["text"] for b in batch],
        "input_widths": torch.tensor(widths, dtype=torch.long),
        "difficulties": [b["difficulty"] for b in batch],
        "sample_types": [b["sample_type"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }


# ============================================================
# 5. MODEL
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    def __init__(self, num_classes: int, cnn_out_channels: int = 512, lstm_hidden: int = 384, lstm_layers: int = 3, dropout: float = 0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1, 64),
            nn.MaxPool2d((2, 2)),

            ConvBlock(64, 128),
            nn.MaxPool2d((2, 2)),

            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),

            ConvBlock(256, 384),
            ConvBlock(384, 384),
            nn.MaxPool2d((2, 1), (2, 1)),

            ConvBlock(384, cnn_out_channels),
            nn.MaxPool2d((4, 1), (4, 1)),
        )

        self.map_to_seq = nn.Linear(cnn_out_channels, lstm_hidden)

        self.rnn = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )

        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)          # [B, C, 1, W']
        feat = feat.squeeze(2)      # [B, C, W']
        feat = feat.permute(2, 0, 1)  # [T, B, C]
        feat = self.map_to_seq(feat)
        seq, _ = self.rnn(feat)
        logits = self.classifier(seq)
        return logits


def compute_output_seq_len(input_widths: torch.Tensor) -> torch.Tensor:
    out = input_widths.clone()
    out = torch.div(out, 2, rounding_mode='floor')
    out = torch.div(out, 2, rounding_mode='floor')
    out = torch.clamp(out, min=1)
    return out


# ============================================================
# 6. DECODE
# ============================================================

@torch.no_grad()
def greedy_decode(logits: torch.Tensor, tokenizer: CTCTokenizer) -> List[str]:
    pred_ids = logits.argmax(dim=-1)
    pred_ids = pred_ids.permute(1, 0).cpu().tolist()

    texts = []
    for seq in pred_ids:
        texts.append(tokenizer.decode_to_text(seq))
    return texts


# ============================================================
# 7. TENSORBOARD HELPERS
# ============================================================

def log_gpu_stats(writer: SummaryWriter, global_step: int):
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        writer.add_scalar("system/gpu_mem_allocated_gb", mem_alloc, global_step)
        writer.add_scalar("system/gpu_mem_reserved_gb", mem_reserved, global_step)
        writer.add_scalar("system/gpu_max_mem_allocated_gb", max_mem_alloc, global_step)


def log_sample_images(writer: SummaryWriter, batch_images: torch.Tensor, tag: str, global_step: int, max_images: int = 8):
    imgs = batch_images[:max_images].detach().cpu()
    writer.add_images(tag, imgs, global_step)


# ============================================================
# 8. EVAL
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: CTCTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    token_mode: str,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    split_name: str = "val",
):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, reduction="sum", zero_infinity=True)

    total_cer = 0.0
    total_wer = 0.0
    exact_match = 0

    difficulty_buckets = {}
    sample_type_buckets = {}
    first_logged_batch = False

    for batch in tqdm(loader, desc=f"Evaluating-{split_name}", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        labels_concat = batch["labels_concat"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        input_widths = batch["input_widths"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)

        input_lengths = compute_output_seq_len(input_widths).to(device)
        loss = ctc_loss(log_probs, labels_concat, input_lengths, target_lengths)

        preds = greedy_decode(logits, tokenizer)
        gts = batch["texts"]

        bs = len(gts)
        total_loss += loss.item()
        total_samples += bs

        if writer is not None and global_step is not None and not first_logged_batch:
            writer.add_text(
                f"{split_name}/predictions",
                format_examples_for_tensorboard(preds, gts, batch["image_paths"], max_examples=8),
                global_step
            )
            log_sample_images(writer, images, f"{split_name}/sample_images", global_step, max_images=8)
            first_logged_batch = True

        for pred, gt, diff, stype in zip(preds, gts, batch["difficulties"], batch["sample_types"]):
            cer = compute_cer(pred, gt, token_mode=token_mode)
            wer = compute_wer(pred, gt)

            total_cer += cer
            total_wer += wer
            exact_match += int(pred == gt)

            if diff not in difficulty_buckets:
                difficulty_buckets[diff] = {"count": 0, "cer": 0.0, "wer": 0.0}
            difficulty_buckets[diff]["count"] += 1
            difficulty_buckets[diff]["cer"] += cer
            difficulty_buckets[diff]["wer"] += wer

            if stype not in sample_type_buckets:
                sample_type_buckets[stype] = {"count": 0, "cer": 0.0, "wer": 0.0}
            sample_type_buckets[stype]["count"] += 1
            sample_type_buckets[stype]["cer"] += cer
            sample_type_buckets[stype]["wer"] += wer

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "cer": total_cer / max(total_samples, 1),
        "wer": total_wer / max(total_samples, 1),
        "exact_match": exact_match / max(total_samples, 1),
        "difficulty": {},
        "sample_type": {},
    }

    for k, v in difficulty_buckets.items():
        metrics["difficulty"][k] = {
            "cer": v["cer"] / v["count"],
            "wer": v["wer"] / v["count"],
            "count": v["count"],
        }

    for k, v in sample_type_buckets.items():
        metrics["sample_type"][k] = {
            "cer": v["cer"] / v["count"],
            "wer": v["wer"] / v["count"],
            "count": v["count"],
        }

    if writer is not None and global_step is not None:
        writer.add_scalar(f"{split_name}/loss", metrics["loss"], global_step)
        writer.add_scalar(f"{split_name}/cer", metrics["cer"], global_step)
        writer.add_scalar(f"{split_name}/wer", metrics["wer"], global_step)
        writer.add_scalar(f"{split_name}/exact_match", metrics["exact_match"], global_step)

        for diff_name, vals in metrics["difficulty"].items():
            writer.add_scalar(f"{split_name}/difficulty_{diff_name}/cer", vals["cer"], global_step)
            writer.add_scalar(f"{split_name}/difficulty_{diff_name}/wer", vals["wer"], global_step)

        for stype_name, vals in metrics["sample_type"].items():
            writer.add_scalar(f"{split_name}/sample_type_{stype_name}/cer", vals["cer"], global_step)
            writer.add_scalar(f"{split_name}/sample_type_{stype_name}/wer", vals["wer"], global_step)

    return metrics


# ============================================================
# 9. TRAIN
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    tokenizer: CTCTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    grad_clip: float,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    log_interval: int = 100,
):
    model.train()

    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, reduction="mean", zero_infinity=True)
    running_loss = 0.0
    total_batches = 0
    start = time.time()

    for step, batch in enumerate(tqdm(loader, desc=f"Training-epoch-{epoch}", leave=False), start=1):
        images = batch["images"].to(device, non_blocking=True)
        labels_concat = batch["labels_concat"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        input_widths = batch["input_widths"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = compute_output_seq_len(input_widths).to(device)
            loss = ctc_loss(log_probs, labels_concat, input_lengths, target_lengths)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        total_batches += 1
        global_step += 1

        writer.add_scalar("train/loss_step", loss.item(), global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
        log_gpu_stats(writer, global_step)

        if step == 1:
            log_sample_images(writer, images, "train/sample_images", global_step, max_images=8)

        if step % log_interval == 0:
            avg_loss = running_loss / total_batches
            elapsed = time.time() - start
            writer.add_scalar("train/loss_running", avg_loss, global_step)
            print(f"[train] epoch={epoch} step={step} avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s")

    epoch_loss = running_loss / max(total_batches, 1)
    writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
    return epoch_loss, global_step


# ============================================================
# 10. MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, default="output/train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="output/val.jsonl")
    parser.add_argument("--test_jsonl", type=str, default="output/test.jsonl")
    parser.add_argument("--vocab_path", type=str, default="output/grapheme_vocab.txt")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--tb_logdir", type=str, default="runs/kaithi_crnn")

    parser.add_argument("--token_mode", type=str, default="grapheme", choices=["char", "grapheme"])
    parser.add_argument("--sample_type", type=str, default="word", choices=["word", "line", "both"])
    parser.add_argument("--img_height", type=int, default=64)
    parser.add_argument("--max_width", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    parser.add_argument("--lstm_hidden", type=int, default=384)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval_only", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_logdir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required for this training pipeline.")

    amp_dtype = torch.bfloat16 if args.use_bf16 else torch.float16

    writer = SummaryWriter(log_dir=args.tb_logdir)

    tokenizer = CTCTokenizer(args.vocab_path)

    train_cfg = DatasetConfig(
        root_dir=args.data_root,
        jsonl_path=args.train_jsonl,
        token_mode=args.token_mode,
        sample_type=args.sample_type,
        img_height=args.img_height,
        max_width=args.max_width,
    )
    val_cfg = DatasetConfig(
        root_dir=args.data_root,
        jsonl_path=args.val_jsonl,
        token_mode=args.token_mode,
        sample_type=args.sample_type,
        img_height=args.img_height,
        max_width=args.max_width,
    )
    test_cfg = DatasetConfig(
        root_dir=args.data_root,
        jsonl_path=args.test_jsonl,
        token_mode=args.token_mode,
        sample_type=args.sample_type,
        img_height=args.img_height,
        max_width=args.max_width,
    )

    train_ds = KaithiOCRDataset(train_cfg, tokenizer)
    val_ds = KaithiOCRDataset(val_cfg, tokenizer)
    test_ds = KaithiOCRDataset(test_cfg, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = CRNN(
        num_classes=tokenizer.vocab_size,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    if args.compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    writer.add_text("config/args", json.dumps(vars(args), indent=2), 0)
    writer.add_text("config/device", str(device), 0)
    writer.add_scalar("dataset/train_size", len(train_ds), 0)
    writer.add_scalar("dataset/val_size", len(val_ds), 0)
    writer.add_scalar("dataset/test_size", len(test_ds), 0)
    writer.add_scalar("dataset/vocab_size", tokenizer.vocab_size, 0)

    start_epoch = 1
    best_val_cer = float("inf")
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_cer = ckpt.get("best_val_cer", best_val_cer)
        global_step = ckpt.get("global_step", 0)

    if args.eval_only:
        val_metrics = evaluate(model, val_loader, tokenizer, device, amp_dtype, args.token_mode, writer, global_step, "val")
        test_metrics = evaluate(model, test_loader, tokenizer, device, amp_dtype, args.token_mode, writer, global_step, "test")
        print("[VAL]", json.dumps(val_metrics, ensure_ascii=False, indent=2))
        print("[TEST]", json.dumps(test_metrics, ensure_ascii=False, indent=2))
        writer.close()
        return

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            log_interval=50,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            token_mode=args.token_mode,
            writer=writer,
            global_step=global_step,
            split_name="val",
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            token_mode=args.token_mode,
            writer=writer,
            global_step=global_step,
            split_name="test",
        )

        print(f"[epoch {epoch}] train_loss={train_loss:.4f}")
        print(f"[epoch {epoch}] val_cer={val_metrics['cer']:.4f} val_wer={val_metrics['wer']:.4f} val_exact={val_metrics['exact_match']:.4f}")
        print(f"[epoch {epoch}] test_cer={test_metrics['cer']:.4f} test_wer={test_metrics['wer']:.4f} test_exact={test_metrics['exact_match']:.4f}")

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict() if not hasattr(model, "_orig_mod") else model._orig_mod.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_cer": best_val_cer,
            "args": vars(args),
        }

        torch.save(ckpt, os.path.join(args.save_dir, "latest.pt"))

        if val_metrics["cer"] < best_val_cer:
            best_val_cer = val_metrics["cer"]
            ckpt["best_val_cer"] = best_val_cer
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

        with open(os.path.join(args.save_dir, "metrics_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val": val_metrics,
                "test": test_metrics,
            }, ensure_ascii=False) + "\n")

    writer.close()


if __name__ == "__main__":
    main()