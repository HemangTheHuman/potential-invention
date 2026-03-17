import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def geometry_to_bbox_pixels(geom, page_w, page_h):
    """
    Convert docTR normalized geometry to pixel bbox.
    Supports:
      1. straight box: ((xmin, ymin), (xmax, ymax))
      2. polygon/rotated box: ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
    """
    if not geom:
        return None

    # straight box
    if len(geom) == 2 and len(geom[0]) == 2 and len(geom[1]) == 2:
        (xmin, ymin), (xmax, ymax) = geom
        return {
            "xmin": int(round(xmin * page_w)),
            "ymin": int(round(ymin * page_h)),
            "xmax": int(round(xmax * page_w)),
            "ymax": int(round(ymax * page_h)),
        }

    # polygon / quadrilateral
    xs = [pt[0] for pt in geom]
    ys = [pt[1] for pt in geom]
    return {
        "xmin": int(round(min(xs) * page_w)),
        "ymin": int(round(min(ys) * page_h)),
        "xmax": int(round(max(xs) * page_w)),
        "ymax": int(round(max(ys) * page_h)),
    }


def extract_pdf_words(exported):
    """
    Flatten docTR export into:
    {
      "pages": [
        {
          "page_index": 0,
          "width": ...,
          "height": ...,
          "words": [...]
        }
      ]
    }
    """
    final_output = {
        "num_pages": len(exported.get("pages", [])),
        "pages": []
    }
    for page_idx, page in tqdm(enumerate(exported.get("pages", [])), total=len(exported.get("pages", []))):
        page_h, page_w = page["dimensions"]   # docTR stores (height, width)
        page_words = []

        for block_idx, block in enumerate(page.get("blocks", [])):
            for line_idx, line in enumerate(block.get("lines", [])):
                for word_idx, word in enumerate(line.get("words", [])):
                    geom = word.get("geometry")
                    bbox = geometry_to_bbox_pixels(geom, page_w, page_h)

                    page_words.append({
                        "id": f"p{page_idx}_b{block_idx}_l{line_idx}_w{word_idx}",
                        "text": word.get("value", ""),
                        "confidence": word.get("confidence"),
                        "geometry_normalized": geom,
                        "bbox_pixels": bbox,
                        "block_index": block_idx,
                        "line_index": line_idx,
                        "word_index": word_idx,
                    })

        final_output["pages"].append({
            "page_index": page_idx,
            "width": page_w,
            "height": page_h,
            "num_words": len(page_words),
            "words": page_words,
        })

    return final_output


def main():
    parser = argparse.ArgumentParser(description="Extract word-level bounding boxes from a PDF using docTR")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", default="pdf_words.json", help="Output JSON path")
    parser.add_argument("--det-arch", default="db_resnet50", help="docTR detection architecture")
    parser.add_argument("--reco-arch", default="crnn_vgg16_bn", help="docTR recognition architecture")
    parser.add_argument("--straight", action="store_true", help="Assume straight pages")
    args = parser.parse_args()

    input_pdf = Path(args.input)
    output_json = Path(args.output)

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ocr_predictor(
        det_arch=args.det_arch,
        reco_arch=args.reco_arch,
        pretrained=True,
        assume_straight_pages=args.straight,
        preserve_aspect_ratio=True,
    ).to(device)

    # Read all PDF pages
    doc = DocumentFile.from_pdf(str(input_pdf))

    print(f"Loaded PDF with {len(doc)} page(s)")

    # OCR
    result = model(doc)

    # Export structured result
    exported = result.export()

    # Convert to simpler JSON
    final_json = extract_pdf_words(exported)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON to: {output_json}")


if __name__ == "__main__":
    main()