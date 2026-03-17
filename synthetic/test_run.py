import os
import torch
import json
from inference import process_json_ocr
from RCNN_trainer import CRNN, CTCTokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    vocab_path = "/home/azureuser/kaithi/synthetic/output/grapheme_vocab.txt"
    ckpt_path = "/home/azureuser/kaithi/synthetic/checkpoints/grapheme_word/best.pt"
    
    if not os.path.exists(ckpt_path):
        # Fallback to grapheme_line if grapheme_word doesn't exist
        ckpt_path = "/home/azureuser/kaithi/synthetic/checkpoints/grapheme_line/best.pt"
        
    print(f"Loading checkpoint: {ckpt_path}")
    tokenizer = CTCTokenizer(vocab_path)
    model = CRNN(num_classes=tokenizer.vocab_size, lstm_hidden=384, lstm_layers=3, dropout=0.2).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("_orig_mod."):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    json_path = "/home/azureuser/kaithi/OCD/data/wordskaithi.json"
    pdf_path = "/home/azureuser/kaithi/OCD/data/test.pdf" 
    output_json_path = "/home/azureuser/kaithi/OCD/data/wordskaithi_out.json"
    
    print("Starting process_json_ocr...")
    process_json_ocr(
        json_path=json_path,
        pdf_path=pdf_path,
        output_json_path=output_json_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
        amp_dtype=amp_dtype,
        batch_size=16,
        num_workers=0,
        max_width=768
    )
    
    with open(output_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        pages = data.get("pages", [])
        if pages and pages[0].get("words"):
            first_word = pages[0]["words"][0]
            print(f"Sample output: {first_word.get('model_output', 'NOT FOUND')}")

if __name__ == "__main__":
    main()
