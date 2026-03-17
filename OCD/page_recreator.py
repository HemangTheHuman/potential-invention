import json
import argparse
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

def recreate_pdf(json_path, output_pdf, font_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Register Kaithi font
    pdfmetrics.registerFont(TTFont('Kaithi', font_path))

    # Initialize canvas
    c = canvas.Canvas(str(output_pdf))

    pages = data.get("pages", [])
    
    for page in pages:
        page_w = page.get("width", 1224)
        page_h = page.get("height", 1584)
        
        # In reportlab, default is bottom-up (y=0 is at bottom).
        c.setPageSize((page_w, page_h))
        
        words = page.get("words", [])
        for word in words:
            kaithi_str = word.get("kaithi", "")
            if not kaithi_str:
                continue
                
            bbox = word.get("bbox_pixels")
            if not bbox:
                continue
            
            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]
            
            # Compute roughly the font size
            # the height of bounding box
            box_h = ymax - ymin
            
            # Subtlety: reportlab coordinate system puts y=0 at the BOTTOM
            # OCR system puts y=0 at the TOP.
            # So y coordinate for text bottom in reportlab is: page_h - ymax
            y_reportlab = page_h - ymax
            x_reportlab = xmin
            
            c.setFont("Kaithi", box_h)
            c.drawString(x_reportlab, y_reportlab + box_h * 0.15, kaithi_str)  # add a small baseline offset
            
        c.showPage()
    
    c.save()
    print(f"Saved generated PDF to: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Recreate a PDF containing Kaithi text from words JSON.")
    parser.add_argument("--json", default="/home/azureuser/kaithi/OCD/data/wordskaithi.json", help="Path to words JSON")
    parser.add_argument("--output", default="/home/azureuser/kaithi/OCD/data/recreated_kaithi.pdf", help="Output PDF path")
    parser.add_argument("--font", default="/home/azureuser/kaithi/synthetic/fonts/NotoSansKaithi-Regular.ttf", help="Path to Kaithi TTF font")
    
    args = parser.parse_args()
    
    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
        
    font_path = Path(args.font)
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")
        
    recreate_pdf(json_path, args.output, font_path)

if __name__ == "__main__":
    main()
