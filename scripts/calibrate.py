r"""
Calibration du seuil de confiance - le levier le moins cher pour gagner du rappel.

    python scripts/calibrate.py --weights best.pt --data datasets/ppe --split val
    python scripts/calibrate.py --weights best.pt --data datasets/ppe --split val --target-recall 0.95

Pourquoi : l'evaluation Ultralytics affiche P/R au seuil qui maximise le F1, pas au seuil
que tu utiliseras en production. En securite, on veut un rappel cible (ex. 95 %), quitte a
accepter plus de fausses alertes. Ce script te dit QUEL seuil choisir pour l'atteindre.

Sortie : tableau conf / P / R / F1 global + par classe, et 2 recommandations :
  - seuil optimal F1 (equilibre)
  - seuil pour atteindre le rappel cible (securite)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAMES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre deux jeux de boites xyxy. Retourne (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match(gt: np.ndarray, pred: np.ndarray, pred_conf: np.ndarray, conf: float) -> tuple[int, int, int]:
    """Appariement glouton 1-1 au seuil `conf`. Retourne (TP, FP, FN)."""
    keep = pred_conf >= conf
    p = pred[keep]
    if len(p) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(p), 0
    iou = iou_matrix(gt, p)
    tp = 0
    used_g, used_p = set(), set()
    # on prend les meilleurs recouvrements d'abord
    order = np.argsort(-iou, axis=None)
    for flat in order:
        gi, pi = np.unravel_index(flat, iou.shape)
        if iou[gi, pi] < 0.5:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1
    return tp, len(p) - tp, len(gt) - tp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--data", default="datasets/ppe")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--target-recall", type=float, default=0.95)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.data).resolve()
    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    model = YOLO(args.weights)

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print(f"[X] aucune image dans {img_dir}")
        return

    # predictions a seuil tres bas : on garde tout, on filtrera nous-memes
    results = model.predict([str(p) for p in imgs], imgsz=args.imgsz,
                            conf=0.001, verbose=False)

    pred_by_img: list[tuple[int, dict[int, tuple[np.ndarray, np.ndarray]]]] = []

    for img, r in zip(imgs, results):
        lbl = lbl_dir / f"{img.stem}.txt"
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
        per_class: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for c in range(len(NAMES)):
                m = cls == c
                per_class[c] = (xyxy[m], cf[m])
        pred_by_img.append((len(lines), per_class))

    thresholds = np.arange(0.05, 0.96, 0.05)

    # --- evaluation robuste : on reconstruit les GT par image
    gt_by_img: list[dict[int, np.ndarray]] = []
    for img in imgs:
        lbl = lbl_dir / f"{img.stem}.txt"
        W = H = None
        d: dict[int, list[list[float]]] = {c: [] for c in range(len(NAMES))}
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
        # dimensions image
        from PIL import Image
        with Image.open(img) as im:
            W, H = im.size
        for parts in lines:
            c = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:])
            d[c].append([(xc - bw / 2) * W, (yc - bh / 2) * H,
                         (xc + bw / 2) * W, (yc + bh / 2) * H])
        gt_by_img.append({c: np.array(v, dtype=np.float32) for c, v in d.items() if v})

    rows = []
    for t in thresholds:
        TP = {c: 0 for c in range(len(NAMES))}
        FP = {c: 0 for c in range(len(NAMES))}
        FN = {c: 0 for c in range(len(NAMES))}
        for gt_d, (_, pc_d) in zip(gt_by_img, pred_by_img):
            for c in range(len(NAMES)):
                gt = gt_d.get(c, np.zeros((0, 4), dtype=np.float32))
                pr, cf = pc_d.get(c, (np.zeros((0, 4), dtype=np.float32),
                                      np.zeros((0,), dtype=np.float32)))
                tp, fp, fn = match(gt, pr, cf, t)
                TP[c] += tp
                FP[c] += fp
                FN[c] += fn
        per_c = {}
        for c in range(len(NAMES)):
            p = TP[c] / max(1, TP[c] + FP[c])
            r = TP[c] / max(1, TP[c] + FN[c])
            f1 = 2 * p * r / max(1e-9, p + r)
            per_c[c] = (p, r, f1)
        TPt, FPt, FNt = sum(TP.values()), sum(FP.values()), sum(FN.values())
        Pt = TPt / max(1, TPt + FPt)
        Rt = TPt / max(1, TPt + FNt)
        F1t = 2 * Pt * Rt / max(1e-9, Pt + Rt)
        rows.append((t, Pt, Rt, F1t, per_c))

    print(f"\n=== Calibration sur {args.split} ({len(imgs)} images, "
          f"{sum(len(v) for d in gt_by_img for v in d.values())} objets) ===")
    print(f"{'conf':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for t, P, R, F1, _ in rows:
        bar = "#" * int(30 * F1)
        print(f"{t:>6.2f}{P:>8.3f}{R:>8.3f}{F1:>8.3f}  {bar}")

    best_f1 = max(rows, key=lambda x: x[3])
    print(f"\n-> Seuil optimal F1     : conf = {best_f1[0]:.2f}  "
          f"(P={best_f1[1]:.3f}  R={best_f1[2]:.3f}  F1={best_f1[3]:.3f})")

    reach = [r for r in rows if r[2] >= args.target_recall]
    if reach:
        t_ok = min(reach, key=lambda x: x[0] + 0)  # le plus haut seuil qui atteint la cible
        print(f"-> Seuil pour R >= {args.target_recall:.0%} : conf = {t_ok[0]:.2f}  "
              f"(P={t_ok[1]:.3f}  R={t_ok[2]:.3f})")
    else:
        print(f"-> Rappel cible {args.target_recall:.0%} INATTEIGNABLE meme a conf=0.05 "
              f"(max R={max(r[2] for r in rows):.3f}) -> il faut ameliorer le modele")

    # --- seuils par classe au rappel cible
    print(f"\n=== Seuil par classe pour R >= {args.target_recall:.0%} ===")
    print(f"{'classe':<16}{'conf':>7}{'P':>8}{'R':>8}")
    for c in range(len(NAMES)):
        ok = [(t, pc[c][0], pc[c][1]) for t, _, _, _, pc in rows if pc[c][1] >= args.target_recall]
        if ok:
            t, P, R = min(ok, key=lambda x: x[0])
            print(f"{NAMES[c]:<16}{t:>7.2f}{P:>8.3f}{R:>8.3f}")
        else:
            best = max(((t, pc[c][0], pc[c][1]) for t, _, _, _, pc in rows), key=lambda x: x[2])
            print(f"{NAMES[c]:<16}{'--':>7}  rappel max = {best[2]:.3f}  (classe limitante)")

    print("\nConseil : utilise ces seuils dans predict.py --conf, ou par classe dans du code custom.")


if __name__ == "__main__":
    main()
