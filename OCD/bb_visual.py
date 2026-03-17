import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from doctr.io import DocumentFile


def load_page_image_from_pdf(pdf_path, page_index):
    """
    Render one PDF page using docTR's PDF loader.
    Returns a PIL image.
    """
    doc = DocumentFile.from_pdf(str(pdf_path))
    if page_index < 0 or page_index >= len(doc):
        raise IndexError(f"Page index {page_index} out of range. PDF has {len(doc)} pages.")

    page = doc[page_index]

    # docTR returns page images as numpy arrays
    if not isinstance(page, Image.Image):
        page = Image.fromarray(page)

    return page


def draw_boxes(page_img, words, show_text=True, line_width=2):
    """
    Draw word boxes from earlier JSON format:
    word["bbox_pixels"] = {xmin, ymin, xmax, ymax}
    """
    img = page_img.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for word in words:
        bbox = word.get("bbox_pixels")
        if not bbox:
            continue

        xmin = bbox["xmin"]
        ymin = bbox["ymin"]
        xmax = bbox["xmax"]
        ymax = bbox["ymax"]

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=line_width)

        if show_text:
            text = word.get("text", "")
            if text:
                text_y = max(0, ymin - 12)
                draw.text((xmin, text_y), text, fill="blue", font=font)

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize docTR word bounding boxes for one PDF page")
    parser.add_argument("--pdf", default="/home/azureuser/kaithi/OCD/data/test.pdf", help="Path to input PDF")
    parser.add_argument("--json", default="/home/azureuser/kaithi/OCD/data/words.json", help="Path to OCR JSON from previous script")
    parser.add_argument("--page", default=0,type=int, required=True, help="0-based page index")
    parser.add_argument("--output", default="/home/azureuser/kaithi/OCD/data/page_boxes.png", help="Output image path")
    parser.add_argument("--hide-text", action="store_true", help="Do not draw recognized text labels")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    json_path = Path(args.json)
    output_path = Path(args.output)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    if args.page < 0 or args.page >= len(pages):
        raise IndexError(f"Page {args.page} not found in JSON. JSON has {len(pages)} pages.")

    page_data = pages[args.page]
    words = page_data.get("words", [])

    page_img = load_page_image_from_pdf(pdf_path, args.page)
    vis = draw_boxes(page_img, words, show_text=not args.hide_text)

    vis.save(output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()