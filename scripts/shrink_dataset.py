r"""
Reduit le poids du dataset sans rien perdre d'utile pour l'entrainement.

    python scripts/shrink_dataset.py --src .\datasets\ppe --dst .\datasets\ppe_light --max-side 800 --quality 88

Pourquoi c'est sans perte reelle :
  - l'entrainement se fait a imgsz=640 : au-dela de 640 px de cote, les pixels
    supplementaires ne sont PAS utilises par le reseau ;
  - les labels YOLO sont normalises (0-1), donc le redimensionnement ne les
    invalide pas : aucun .txt n'est modifie ;
  - le mosaic recadre et redimensionne de toute facon les images.

Resultat typique : 418 Mo -> 60-90 Mo (-80 %), soit ~25 min d'upload au lieu de 2 h 30.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

NAMES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]
EXTS = {".jpg", ".jpeg", ".png"}


def human(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r".\datasets\ppe")
    ap.add_argument("--dst", default=r".\datasets\ppe_light")
    ap.add_argument("--max-side", type=int, default=800,
                    help="cote max en pixels. 640 = taille d'entrainement, 800 = marge de securite")
    ap.add_argument("--quality", type=int, default=88, help="qualite JPEG (85-92 = bon compromis)")
    ap.add_argument("--zip", default=None, help="cree aussi un zip (chemin de sortie)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not (src / "images").is_dir():
        print(f"[X] {src / 'images'} introuvable")
        return 1

    before = dir_size(src / "images")

    if dst.exists():
        print(f"[i] suppression de l'ancien {dst}")
        shutil.rmtree(dst)

    n_img = 0
    for split in ("train", "val", "test"):
        img_in = src / "images" / split
        if not img_in.is_dir():
            continue
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

        # labels : copies a l'identique (coordonnees normalisees -> rien a recalculer)
        for lbl in (src / "labels" / split).glob("*.txt"):
            shutil.copy2(lbl, dst / "labels" / split / lbl.name)

        for img_path in sorted(img_in.iterdir()):
            if img_path.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    scale = args.max_side / max(w, h)
                    if scale < 1.0:
                        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                       Image.LANCZOS)
                    im.save(dst / "images" / split / (img_path.stem + ".jpg"),
                            "JPEG", quality=args.quality, optimize=True, progressive=True)
                    n_img += 1
            except Exception as e:  # noqa: BLE001
                print(f"[!] {img_path.name} ignore : {e}")
        print(f"  {split:<6} {(dst / 'images' / split)} -> {len(list((dst / 'images' / split).iterdir()))} images")

    after = dir_size(dst / "images")

    (dst / "ppe.yaml").write_text(
        f"path: {dst.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(NAMES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )

    print(f"\n=== Redimensionnement termine ===")
    print(f"images traitees : {n_img}")
    print(f"avant : {human(before)}   apres : {human(after)}   "
          f"(-{100 * (1 - after / max(1, before)):.0f} %)")

    if args.zip:
        zpath = Path(args.zip)
        if zpath.exists():
            zpath.unlink()
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(dst.rglob("*")):
                if f.is_file():
                    # IMPORTANT : arcname avec '/' (norme ZIP) -> Kaggle accepte
                    z.write(f, f.relative_to(dst).as_posix())
        size = zpath.stat().st_size
        print(f"\nzip : {zpath}  ({human(size)})")
        for mbps in (0.39, 1, 5, 10):
            minutes = (size * 8 / 1e6) / mbps / 60
            print(f"  upload a {mbps} Mb/s -> {minutes:.0f} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
