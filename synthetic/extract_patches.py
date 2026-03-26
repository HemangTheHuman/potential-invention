import os
import cv2
import fitz
import glob
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================================================
# Helpers
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_pdfs(inp_dir):
    files = []
    for ext in ("*.pdf", "*.PDF"):
        files.extend(glob.glob(os.path.join(inp_dir, "**", ext), recursive=True))
    return sorted(files)


def detect_lines(gray, horizontal=True):
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 11
    )

    if horizontal:
        klen = max(20, gray.shape[1] // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    else:
        klen = max(20, gray.shape[0] // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))

    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)


def build_foreground_masks(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
    diff = cv2.absdiff(gray, bg)

    _, dark_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY_INV)
    _, diff_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)

    ink_mask = cv2.bitwise_or(dark_mask, diff_mask)

    kernel = np.ones((3, 3), np.uint8)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(gray, 60, 150)

    horiz = detect_lines(gray, horizontal=True)
    vert = detect_lines(gray, horizontal=False)
    line_mask = cv2.bitwise_or(horiz, vert)

    return gray, ink_mask, edges, line_mask


def component_count(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    count = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if 5 <= area <= 5000:
            count += 1
    return count


def mean_color_std(patch):
    return float(np.std(patch.reshape(-1, 3), axis=0).mean())


def classify_patch(patch_bgr, ink_ratio, edge_ratio, line_ratio):
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    brightness_std = float(np.std(gray))

    if line_ratio < 0.01 and ink_ratio < 0.02 and edge_ratio < 0.03 and brightness_std < 18:
        return "clean_paper"

    if brightness_std >= 18 or (0.01 <= ink_ratio < 0.035):
        return "stained_paper"

    return "edge_and_shadow"


def is_near_page_border(x, y, w, h, img_w, img_h, border_frac=0.08):
    bx = int(img_w * border_frac)
    by = int(img_h * border_frac)
    return x < bx or y < by or (x + w) > (img_w - bx) or (y + h) > (img_h - by)


def score_window(ink_patch, edge_patch, line_patch, color_patch, prefer_border=False):
    area = ink_patch.shape[0] * ink_patch.shape[1]
    ink_ratio = float(np.count_nonzero(ink_patch)) / area
    edge_ratio = float(np.count_nonzero(edge_patch)) / area
    line_ratio = float(np.count_nonzero(line_patch)) / area
    comp_cnt = component_count(ink_patch)
    color_std = mean_color_std(color_patch)

    score = (
        ink_ratio * 8.0 +
        edge_ratio * 3.5 +
        line_ratio * 6.0 +
        min(comp_cnt / 40.0, 1.5) * 2.5
    )

    if 4 <= color_std <= 28:
        score -= 0.15
    if prefer_border:
        score -= 0.10

    return score, {
        "ink_ratio": ink_ratio,
        "edge_ratio": edge_ratio,
        "line_ratio": line_ratio,
        "comp_cnt": comp_cnt,
        "color_std": color_std,
    }


def non_max_suppression_boxes(cands, iou_thr=0.25):
    if not cands:
        return []

    cands = sorted(cands, key=lambda z: z["score"])
    picked = []

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / union if union > 0 else 0.0

    for c in cands:
        keep = True
        for p in picked:
            if iou(c, p) > iou_thr:
                keep = False
                break
        if keep:
            picked.append(c)

    return picked


def render_pdf_page(pdf_path, page_idx, dpi):
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    finally:
        doc.close()


def draw_debug_preview(img_bgr, candidates, save_path):
    vis = img_bgr.copy()
    for c in candidates:
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(save_path, vis)


# =========================================================
# Core extraction from one page
# =========================================================

def extract_candidate_patches_from_page(
    img_bgr,
    source_id,
    output_dir,
    patch_sizes=(256, 384, 512),
    max_patches_per_page=8,
    stride_frac=0.45,
    max_candidates_per_size=12,
    save_debug=False,
):
    img_h, img_w = img_bgr.shape[:2]
    gray, ink_mask, edges, line_mask = build_foreground_masks(img_bgr)

    all_candidates = []

    for psize in patch_sizes:
        if psize >= img_w or psize >= img_h:
            continue

        stride = max(32, int(psize * stride_frac))
        local_candidates = []

        for y in range(0, img_h - psize + 1, stride):
            for x in range(0, img_w - psize + 1, stride):
                ink_patch = ink_mask[y:y+psize, x:x+psize]
                edge_patch = edges[y:y+psize, x:x+psize]
                line_patch = line_mask[y:y+psize, x:x+psize]
                color_patch = img_bgr[y:y+psize, x:x+psize]

                prefer_border = is_near_page_border(x, y, psize, psize, img_w, img_h)
                score, meta = score_window(
                    ink_patch=ink_patch,
                    edge_patch=edge_patch,
                    line_patch=line_patch,
                    color_patch=color_patch,
                    prefer_border=prefer_border
                )

                if meta["ink_ratio"] > 0.045:
                    continue
                if meta["edge_ratio"] > 0.08:
                    continue
                if meta["line_ratio"] > 0.06:
                    continue
                if meta["comp_cnt"] > 35:
                    continue

                local_candidates.append({
                    "x": x,
                    "y": y,
                    "w": psize,
                    "h": psize,
                    "score": score,
                    "meta": meta,
                })

        local_candidates = sorted(local_candidates, key=lambda z: z["score"])[:max_candidates_per_size]
        local_candidates = non_max_suppression_boxes(local_candidates, iou_thr=0.22)
        all_candidates.extend(local_candidates)

    all_candidates = sorted(all_candidates, key=lambda z: z["score"])
    all_candidates = non_max_suppression_boxes(all_candidates, iou_thr=0.20)
    all_candidates = all_candidates[:max_patches_per_page]

    saved = []

    for idx, cand in enumerate(all_candidates):
        x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
        patch = img_bgr[y:y+h, x:x+w].copy()

        bucket = classify_patch(
            patch_bgr=patch,
            ink_ratio=cand["meta"]["ink_ratio"],
            edge_ratio=cand["meta"]["edge_ratio"],
            line_ratio=cand["meta"]["line_ratio"]
        )

        bucket_dir = os.path.join(output_dir, bucket)
        ensure_dir(bucket_dir)

        name = f"{source_id}_x{x}_y{y}_s{w}_{idx:03d}.jpg"
        out_path = os.path.join(bucket_dir, name)
        cv2.imwrite(out_path, patch, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        saved.append({
            "path": out_path,
            "bucket": bucket,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            **cand["meta"],
        })

    if save_debug:
        debug_dir = os.path.join(output_dir, "_debug")
        ensure_dir(debug_dir)
        draw_debug_preview(img_bgr, all_candidates, os.path.join(debug_dir, f"{source_id}_debug.jpg"))

    return saved


# =========================================================
# Worker
# =========================================================

def process_one_page(task):
    (
        pdf_path,
        page_idx,
        output_dir,
        dpi,
        patch_sizes,
        max_patches_per_page,
        stride_frac,
        max_candidates_per_size,
        save_debug,
    ) = task

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    source_id = f"{pdf_name}_p{page_idx+1:04d}"

    try:
        img = render_pdf_page(pdf_path, page_idx, dpi)
        saved = extract_candidate_patches_from_page(
            img_bgr=img,
            source_id=source_id,
            output_dir=output_dir,
            patch_sizes=patch_sizes,
            max_patches_per_page=max_patches_per_page,
            stride_frac=stride_frac,
            max_candidates_per_size=max_candidates_per_size,
            save_debug=save_debug,
        )
        return {"ok": True, "source_id": source_id, "saved": saved}
    except Exception as e:
        return {"ok": False, "source_id": source_id, "error": str(e)}


# =========================================================
# Task building
# =========================================================

def build_page_tasks(
    pdf_dir,
    output_dir,
    dpi,
    patch_sizes,
    max_patches_per_page,
    stride_frac,
    max_candidates_per_size,
    save_debug,
):
    pdf_files = list_pdfs(pdf_dir)
    if not pdf_files:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    tasks = []
    for pdf_path in pdf_files:
        doc = fitz.open(pdf_path)
        try:
            n_pages = len(doc)
        finally:
            doc.close()

        for page_idx in range(n_pages):
            tasks.append((
                pdf_path,
                page_idx,
                output_dir,
                dpi,
                patch_sizes,
                max_patches_per_page,
                stride_frac,
                max_candidates_per_size,
                save_debug,
            ))
    return tasks


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--patch_sizes", type=str, default="256,384,512")
    ap.add_argument("--max_patches_per_page", type=int, default=8)
    ap.add_argument("--stride_frac", type=float, default=0.45)
    ap.add_argument("--max_candidates_per_size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--save_debug", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "clean_paper"))
    ensure_dir(os.path.join(args.output_dir, "stained_paper"))
    ensure_dir(os.path.join(args.output_dir, "edge_and_shadow"))
    if args.save_debug:
        ensure_dir(os.path.join(args.output_dir, "_debug"))

    patch_sizes = tuple(int(x.strip()) for x in args.patch_sizes.split(",") if x.strip())

    tasks = build_page_tasks(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        patch_sizes=patch_sizes,
        max_patches_per_page=args.max_patches_per_page,
        stride_frac=args.stride_frac,
        max_candidates_per_size=args.max_candidates_per_size,
        save_debug=args.save_debug,
    )

    manifest = []
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_page, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing pages"):
            res = fut.result()
            if res["ok"]:
                manifest.extend(res["saved"])
            else:
                errors.append(res)

    manifest_path = os.path.join(args.output_dir, "patch_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if errors:
        with open(os.path.join(args.output_dir, "errors.json"), "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)

    print(f"\nDone. Saved {len(manifest)} patches")
    print(f"Manifest: {manifest_path}")
    if errors:
        print(f"Errors: {len(errors)} pages failed")


if __name__ == "__main__":
    main()