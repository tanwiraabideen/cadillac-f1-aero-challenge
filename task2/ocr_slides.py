"""
Task 2, step 3 of 4: read the text on each distinct screen capture with OCR.

Input:  task2/outputs/slides.csv   (one row per distinct capture, from build_timeline.py)
Output: task2/outputs/slide_ocr.csv         slide_id, meeting, image_file, ocr_text, n_words, mean_conf
        task2/outputs/slide_ocr_words.json  every word kept, with its position on the slide

How OCR works here: the captures are pictures (pixels), not documents with
stored text. Tesseract finds lines of text and recognises the letter shapes,
giving each word a box and a confidence (0-100). Small, blurry JPEG text is the
main risk, so each image is enlarged and turned greyscale first, and
low-confidence words are dropped. Check a few slides by eye against the output.

Word positions are stored as fractions of the image (0 = left/top, 1 = right/bottom),
so "up there" or "on the left" can later be matched to regions whatever the
image size.

Pictures, diagrams and the spreadsheet are covered by slide_notes.csv (written
by hand); OCR on those will be partial or empty, which is expected.

Needs: brew install tesseract   and   pip install pytesseract

    python task2/ocr_slides.py
"""
import csv
import json

import pytesseract
from PIL import Image, ImageOps

# inspect_slides first: it adds task1/ to the import path.
from inspect_slides import OUT_DIR, slide_dir

UPSCALE = 3          # enlarge small slide text before recognising it
MIN_CONF = 60        # drop words Tesseract is less sure of than this (0-100)


def ocr_image(path):
    """Return the kept words: [{text, conf, x, y, w, h, line}] with positions as fractions."""
    with Image.open(path) as im:
        W, H = im.size
        big = ImageOps.grayscale(im).resize((W * UPSCALE, H * UPSCALE), Image.LANCZOS)
    d = pytesseract.image_to_data(big, output_type=pytesseract.Output.DICT)

    words = []
    for i, text in enumerate(d["text"]):
        text = text.strip()
        conf = float(d["conf"][i])
        if not text or conf < MIN_CONF:
            continue
        words.append({
            "text": text,
            "conf": round(conf, 1),
            "x": round(d["left"][i] / (W * UPSCALE), 4),
            "y": round(d["top"][i] / (H * UPSCALE), 4),
            "w": round(d["width"][i] / (W * UPSCALE), 4),
            "h": round(d["height"][i] / (H * UPSCALE), 4),
            # same block/paragraph/line numbers = same line of text on the slide
            "line": f'{d["block_num"][i]}.{d["par_num"][i]}.{d["line_num"][i]}',
        })
    return words


def words_to_text(words):
    """Rebuild readable text, one line of the slide per line."""
    lines, order = {}, []
    for w in words:
        if w["line"] not in lines:
            lines[w["line"]] = []
            order.append(w["line"])
        lines[w["line"]].append(w["text"])
    return "\n".join(" ".join(lines[k]) for k in order)


def main():
    slides_csv = OUT_DIR / "slides.csv"
    if not slides_csv.exists():
        raise SystemExit("Run build_timeline.py first: it writes slides.csv")
    with open(slides_csv) as f:
        slides = list(csv.DictReader(f))

    rows, all_words = [], {}
    for s in slides:
        path = slide_dir(s["meeting"]) / s["image_file"]
        words = ocr_image(path)
        all_words[s["slide_id"]] = words
        mean_conf = sum(w["conf"] for w in words) / len(words) if words else 0
        rows.append({"slide_id": s["slide_id"], "meeting": s["meeting"], "image_file": s["image_file"],
                     "ocr_text": words_to_text(words), "n_words": len(words),
                     "mean_conf": round(mean_conf, 1)})
        preview = words_to_text(words).replace("\n", " | ")[:70]
        print(f"{s['slide_id']:<14} {len(words):3d} words  conf {mean_conf:5.1f}  {preview}")

    with open(OUT_DIR / "slide_ocr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "slide_ocr_words.json").write_text(json.dumps(all_words, indent=1))

    empty = [r["slide_id"] for r in rows if r["n_words"] == 0]
    print(f"\n{len(rows)} captures read -> slide_ocr.csv, slide_ocr_words.json")
    print(f"no text found on {len(empty)}: {empty}")


if __name__ == "__main__":
    main()