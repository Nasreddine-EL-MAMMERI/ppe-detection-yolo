"""
Etape 2 (bonus) - Verification du dataset AVANT entrainement.

    python scripts/dataset_stats.py --data datasets/ppe

Controle :
  - desequilibre des classes (person est souvent 3x plus frequent que vest)
  - taille des boites (les casques sont petits -> justifie imgsz=640)
  - images sans annotation (faux negatifs qui pourrissent l'apprentissage)
  - resolution des images
Sans dependance a ultralytics : lit juste images/ + labels/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/ppe")
    args = ap.parse_args()
    root = Path(args.data).expanduser().resolve()

    for split in ("train", "val", "test"):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.is_dir():
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        counts = Counter()
        empty, boxes_per_img, areas = [], [], defaultdict(list)

        for img in imgs:
            lbl = lbl_dir / f"{img.stem}.txt"
            lines = [l for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            if not lines:
                empty.append(img.name)
                continue
            boxes_per_img.append(len(lines))
            for line in lines:
                p = line.split()
                counts[int(p[0])] += 1
                areas[int(p[0])].append(float(p[3]) * float(p[4]))  # w*h normalise

        total = sum(counts.values()) or 1
        print(f"\n{'=' * 58}\n{split.upper():<10} {len(imgs)} images | {total} objets "
              f"| {total / max(1, len(imgs)):.2f} objets/image")
        print(f"{'=' * 58}")
        for i, name in enumerate(CLASS_NAMES):
            c = counts[i]
            bar = "#" * int(40 * c / max(counts.values() or [1]))
            side = ""
            if c < 0.3 * (total / len(CLASS_NAMES)):
                side = "  <-- sous-representee"
            print(f"  {name:<14}{c:>6} ({100 * c / total:4.1f} %) {bar}{side}")
        if any(areas.values()):
            print("  surface mediane des boites (fraction de l'image) :")
            for i, name in enumerate(CLASS_NAMES):
                if areas[i]:
                    a = sorted(areas[i])[len(areas[i]) // 2]
                    print(f"    {name:<14}{a * 100:5.2f} %  ~{int(a ** 0.5 * 640)} px de cote a imgsz=640")
        if empty:
            print(f"  [!] {len(empty)} image(s) sans aucune boite (ex: {empty[:3]})")
            print("      -> soit de vraies images vides (ok), soit des annotations oubliees (a corriger)")

    print("\nLecture : si une classe fait < 30 % de la moyenne, utilise copy_paste=0.3 ou")
    print("duplique/annote davantage d'exemples. Les casques sont petits -> imgsz=640 > 416.")


if __name__ == "__main__":
    main()
