r"""Verification visuelle des annotations YOLO - AUCUN entrainement, ~5 secondes.

Place ce fichier dans C:\Users\elhad\ppe_project\verif.py (ou dans scripts\).
Puis, depuis C:\Users\elhad\ppe_project :

    python verif.py --data .\datasets\ppe --split train --n 12

Produit verif_train.png : une planche de 12 images avec les boites et les classes.
A regarder : un casque rouge annote helmet_blue (ordre des classes decale),
une personne sans boite (annotation oubliee), des boites decalees (normalisation).
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAMES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]
COLORS = [(255, 64, 64), (64, 255, 64), (64, 128, 255),
          (255, 64, 255), (255, 255, 255), (255, 255, 64)]
EXTS = {".jpg", ".jpeg", ".png"}


def font(size):
    for n in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_one(img_path, lbl_path, cell):
    im = Image.open(img_path).convert("RGB")
    w0, h0 = im.size
    sc = cell / max(w0, h0)
    im = im.resize((int(w0 * sc), int(h0 * sc)))
    W, H = im.size
    d = ImageDraw.Draw(im)
    f = font(max(11, cell // 30))

    lines = lbl_path.read_text().splitlines() if lbl_path.exists() else []
    n = 0
    for line in lines:
        p = line.split()
        if len(p) != 5:
            continue
        cid = int(p[0])
        xc, yc, w, h = (float(v) for v in p[1:])
        x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
        x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
        col = COLORS[cid % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        label = NAMES[cid] if cid < len(NAMES) else "?" + str(cid)
        tw = d.textlength(label, font=f)
        tx = min(max(0, x1), max(0, W - tw - 4))
        ty = y1 - f.size - 2 if y1 - f.size - 2 >= 0 else y1
        d.rectangle([tx, ty, tx + tw + 3, ty + f.size + 2], fill=col)
        d.text((tx + 1, ty), label, fill=(0, 0, 0) if cid > 3 else (255, 255, 255), font=f)
        n += 1

    d.text((4, H - f.size - 4), img_path.name + " | " + str(n) + " obj",
           fill=(255, 255, 0), font=f)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./datasets/ppe")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=420)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.data).resolve()
    imgs_dir = root / "images" / a.split
    lbls_dir = root / "labels" / a.split
    if not imgs_dir.is_dir():
        print("[X] introuvable :", imgs_dir)
        return

    imgs = sorted(p for p in imgs_dir.iterdir() if p.suffix.lower() in EXTS)
    if not imgs:
        print("[X] aucune image dans", imgs_dir)
        return
    random.Random(a.seed).shuffle(imgs)
    imgs = imgs[: a.n]

    cells = [draw_one(p, lbls_dir / (p.stem + ".txt"), a.cell) for p in imgs]
    cols = min(a.cols, len(cells))
    rows = (len(cells) + cols - 1) // cols
    cw, ch = cells[0].size
    sheet = Image.new("RGB", (cols * cw, rows * ch), (25, 25, 25))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * cw, (i // cols) * ch))

    out = Path("verif_" + a.split + ".png")
    sheet.save(out)
    print("[OK]", len(cells), "images ->", out.resolve())
    print("Verifie : couleurs des casques, personnes sans boite, boites decalees.")


if __name__ == "__main__":
    main()
