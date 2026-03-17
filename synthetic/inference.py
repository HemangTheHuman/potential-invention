import os
import json
import time
import random
import argparse
import numpy as np
import torch
from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
from doctr.io import DocumentFile

from RCNN_trainer import CRNN, CTCTokenizer, greedy_decode

class JSONCropDataset(Dataset):
    def __init__(self, pages, words_meta, target_h=64, max_width=768):
        self.pages = pages
        self.words_meta = words_meta
        self.target_h = target_h
        self.max_width = max_width

    def __len__(self):
        return len(self.words_meta)

    def __getitem__(self, idx):
        meta = self.words_meta[idx]
        page_idx = meta["page_idx"]
        bbox = meta["bbox"]
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        
        img = self.pages[page_idx]
        
        # Crop the image using the bounding box
        cropped = img.crop((xmin, ymin, xmax, ymax))
        arr = np.array(cropped)
        
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            x = torch.zeros((1, self.target_h, 32), dtype=torch.float32)
            return x

        scale = self.target_h / h
        new_w = max(32, int(w * scale))
        new_w = min(new_w, self.max_width)
        
        resized = Image.fromarray(arr).resize((new_w, self.target_h), Image.BILINEAR)
        arr_resized = np.array(resized)
        
        x = torch.from_numpy(arr_resized).float().unsqueeze(0) / 255.0  # [1, H, W]
        return x

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

def process_json_ocr(json_path, pdf_path, output_json_path, model, tokenizer, device, amp_dtype, batch_size=16, num_workers=4, max_width=768):
    print(f"Loading PDF from {pdf_path}")
    doc = DocumentFile.from_pdf(pdf_path)
    pages = [Image.fromarray(page).convert("L") for page in doc]
    
    print(f"Loading JSON from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    words_meta = []
    word_refs = []
    
    for page_idx, page_data in enumerate(data.get("pages", [])):
        for word in page_data.get("words", []):
            if "bbox_pixels" in word:
                words_meta.append({
                    "page_idx": page_idx,
                    "bbox": word["bbox_pixels"]
                })
                word_refs.append(word)
                
    dataset = JSONCropDataset(pages, words_meta, target_h=64, max_width=max_width)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_inference,
        pin_memory=(device.type == "cuda")
    )
    
    print(f"Running OCR on {len(words_meta)} words...")
    
    model.eval()
    all_preds = []
    
    start_time = time.time()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            logits = model(batch)
            preds = greedy_decode(logits, tokenizer)
            all_preds.extend(preds)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    # Update JSON
    for i, pred in enumerate(all_preds):
        word_refs[i]["model_output"] = pred
        
    print(f"Saving output to {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/azureuser/kaithi/synthetic/checkpoints/grapheme_line/best.pt")
    parser.add_argument("--jsonl", type=str, default="/home/azureuser/kaithi/synthetic/output/test.jsonl")
    parser.add_argument("--vocab", type=str, default="/home/azureuser/kaithi/synthetic/output/grapheme_vocab.txt")
    parser.add_argument("--image_dir", type=str, default="/home/azureuser/kaithi/synthetic/output")
    parser.add_argument("--image_type", type=str, default="word")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = CTCTokenizer(args.vocab)
    
    # Init model configuration (using defaults from training command/script)
    model = CRNN(num_classes=tokenizer.vocab_size, lstm_hidden=384, lstm_layers=3, dropout=0.2).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint["model"]
    # Handle compiled models prefix if present
    if list(state_dict.keys())[0].startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # Select random image
    with open(args.jsonl, "r", encoding="utf-8") as f:
        rows = []
        for line in f:
            data = json.loads(line)
            if data.get("sample_type") == args.image_type:
                rows.append(data)
    
    sample = random.choice(rows)
    image_rel_path = sample["image_path"]
    actual_text = sample["text"]
    src_text=sample["src_text"]
    
    img_path = os.path.join(args.image_dir, image_rel_path)
    print(f"Randomly selected image: {img_path}")
    
    # Preprocess image
    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    
    h, w = arr.shape[:2]
    target_h = 64
    scale = target_h / h
    new_w = max(32, int(w * scale))
    if args.image_type == "word":
        new_w = min(new_w, 768)  # user training command max_width
    elif args.image_type == "line":
        new_w = min(new_w, 1024)  # user training command max_width
    
    resized = Image.fromarray(arr).resize((new_w, target_h), Image.BILINEAR)
    arr_resized = np.array(resized)
    
    x = torch.from_numpy(arr_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    x = x.to(device)
    
    # Inference
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # Run once to warmup (important for accurate timing with cudnn/autocast)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            logits = model(x)
            preds = greedy_decode(logits, tokenizer)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    
    print("-" * 50)
    print(f"Predicted text: {preds[0]}")
    print(f"Actual text:    {actual_text}")
    print(f"Source text:    {src_text}")
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
    print("-" * 50)

if __name__ == "__main__":
    main()
