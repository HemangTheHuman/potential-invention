import os
import json
import time
import random
import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from RCNN_trainer import CRNN, CTCTokenizer, greedy_decode

def preprocess_image(img_path: str, max_width: int, target_h: int = 64) -> torch.Tensor:
    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    
    h, w = arr.shape[:2]
    scale = target_h / h
    new_w = max(32, int(w * scale))
    new_w = min(new_w, max_width)
    
    resized = Image.fromarray(arr).resize((new_w, target_h), Image.BILINEAR)
    arr_resized = np.array(resized)
    
    x = torch.from_numpy(arr_resized).float().unsqueeze(0) / 255.0  # [1, H, W]
    return x

class InferenceDataset(Dataset):
    def __init__(self, samples, image_dir, max_width):
        self.samples = samples
        self.image_dir = image_dir
        self.max_width = max_width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["image_path"])
        return preprocess_image(img_path, self.max_width)


def collate_inference(tensors):
    widths = [t.shape[-1] for t in tensors]
    max_w = max(widths)
    B = len(tensors)
    H = tensors[0].shape[-2]
    
    x_batch = torch.ones((B, 1, H, max_w), dtype=torch.float32)
    for j, t in enumerate(tensors):
        w = t.shape[-1]
        x_batch[j, :, :, :w] = t
    return x_batch


def run_benchmark(model, tokenizer, rows, args, device, amp_dtype, batch_size=1, compile_model=False, num_workers=0):
    # Select random samples
    samples = random.sample(rows, min(args.num_samples, len(rows)))
    if args.image_type == "word":
        max_w = 768
    elif args.image_type == "line":
        max_w = 1024
    dataset = InferenceDataset(samples, args.image_dir, max_w)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_inference,
        pin_memory=(device.type == "cuda")
    )
    
    # Optional compile
    if compile_model:
        print("Compiling model for inference...")
        model = torch.compile(model, mode="reduce-overhead")
    
    model.eval()
    
    # Warmup
    print("Warming up...")
    x_warmup = dataset[0].unsqueeze(0).to(device)
    for _ in range(5):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            _ = model(x_warmup)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    print(f"Running benchmark on {len(samples)} images (batch_size={batch_size}, workers={num_workers}, compile={compile_model})...")
    
    start_time = time.time()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            logits = model(batch)
            _ = greedy_decode(logits, tokenizer)
                
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(samples)
    
    print("-" * 50)
    print(f"Mode: Batched (BS={batch_size}) | Workers: {num_workers} | Compile: {compile_model}")
    print(f"Total time for {len(samples)} images: {total_time:.4f} "
          f"seconds\nAverage time per image: {avg_time * 1000:.2f} ms")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/azureuser/kaithi/synthetic/checkpoints/grapheme_word/best.pt")
    parser.add_argument("--jsonl", type=str, default="/home/azureuser/kaithi/synthetic/output/test.jsonl")
    parser.add_argument("--vocab", type=str, default="/home/azureuser/kaithi/synthetic/output/grapheme_vocab.txt")
    parser.add_argument("--image_dir", type=str, default="/home/azureuser/kaithi/synthetic/output")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--image_type", type=str, default="word")
    parser.add_argument("--max_width", type=int, default=768)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = CTCTokenizer(args.vocab)
    model = CRNN(num_classes=tokenizer.vocab_size, lstm_hidden=384, lstm_layers=3, dropout=0.2).to(device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    with open(args.jsonl, "r", encoding="utf-8") as f:
        rows = []
        for line in f:
            data = json.loads(line)
            if data.get("sample_type") == args.image_type:
                rows.append(data)

    print("=== Batched (BS=16), 0 Workers, Uncompiled ===")
    run_benchmark(model, tokenizer, rows, args, device, amp_dtype, batch_size=16, compile_model=False, num_workers=0)

    print("=== Batched (BS=16), 4 Workers, Uncompiled ===")
    run_benchmark(model, tokenizer, rows, args, device, amp_dtype, batch_size=16, compile_model=False, num_workers=4)
    
    print("=== Batched (BS=16), 8 Workers, Uncompiled ===")
    run_benchmark(model, tokenizer, rows, args, device, amp_dtype, batch_size=16, compile_model=False, num_workers=8)
    
    print("=== Batched (BS=16), 8 Workers, Compiled ===")
    run_benchmark(model, tokenizer, rows, args, device, amp_dtype, batch_size=16, compile_model=True, num_workers=8)


if __name__ == "__main__":
    main()
