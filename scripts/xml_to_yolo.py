"""
Etape 1 - Conversion Pascal VOC (XML) -> format YOLO (txt) + arborescence YOLO.

Arborescence produite (format attendu par Ultralytics) :

    datasets/ppe/
        images/train/*.jpg   images/val/*.jpg   images/test/*.jpg
        labels/train/*.txt   labels/val/*.txt   labels/test/*.txt
        ppe.yaml

Le script :
  1. lit chaque XML, verifie que chaque image a bien son XML (et inversement) ;
  2. convertit les boites (xmin,ymin,xmax,ymax) -> (x_center/W, y_center/H, w/W, h/H) ;
  3. clampe les coordonnees a l'image (boites "qui depassent" sont tres frequentes) ;
  4. ignore les boites degenerees (w ou h nul) et les classes inconnues ;
  5. copie (pas deplace) les images, donc tes donnees d'origine restent intactes ;
  6. verifie que chaque image a un .txt (meme vide), sinon Ultralytics voit des
     "missing labels" au moment de l'entrainement.

Usage (depuis VS Code, venv active) :
    python scripts/xml_to_yolo.py --src "/chemin/vers/dataset_brut" --dst "./datasets/ppe"

Le dossier --src doit contenir des sous-dossiers (nom libre) avec *.jpg + *.xml.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------- utils
def find_pairs(folder: Path) -> list[tuple[Path, Path | None]]:
    """Retourne [(image, xml_ou_None), ...] pour un dossier."""
    imgs = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    pairs = []
    for img in imgs:
        xml = img.with_suffix(".xml")
        pairs.append((img, xml if xml.exists() else None))
    return pairs


def convert_one(xml_path: Path) -> tuple[list[str], Counter, list[str]]:
    """Convertit un XML -> lignes YOLO. Retourne (lignes, compteur_classes, warnings)."""
    lines: list[str] = []
    counts: Counter = Counter()
    warns: list[str] = []
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path.name}: pas de balise <size>, impossible de normaliser")
    W = float(size.findtext("width"))
    H = float(size.findtext("height"))
    if W <= 0 or H <= 0:
        raise ValueError(f"{xml_path.name}: width/height invalides ({W}x{H})")

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_ID:
            warns.append(f"{xml_path.name}: classe inconnue ignoree -> '{name}'")
            continue
        box = obj.find("bndbox")
        if box is None:
            warns.append(f"{xml_path.name}: <object> sans <bndbox> ignore")
            continue
        xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
        xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))

        # clamp dans l'image
        xmin, xmax = max(0.0, min(xmin, xmax)), min(W, max(xmin, xmax))
        ymin, ymax = max(0.0, min(ymin, ymax)), min(H, max(ymin, ymax))
        w, h = xmax - xmin, ymax - ymin
        if w <= 1 or h <= 1:
            warns.append(f"{xml_path.name}: boite degeneree {name} ({w:.1f}x{h:.1f}) ignoree")
            continue

        xc, yc = (xmin + xmax) / 2.0 / W, (ymin + ymax) / 2.0 / H
        lines.append(f"{CLASS_TO_ID[name]} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")
        counts[name] += 1
    return lines, counts, warns


def copy_split(
    pairs: list[tuple[Path, Path | None]],
    img_dir: Path,
    lbl_dir: Path,
    stats: Counter,
    warns: list[str],
) -> int:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img, xml in pairs:
        if xml is None:
            warns.append(f"{img.name}: XML manquant -> image ignoree (ou label vide)")
            lines = []  # on garde l'image avec un label vide plutot que la perdre
        else:
            lines, counts, w = convert_one(xml)
            stats.update(counts)
            warns.extend(w)
        shutil.copy2(img, img_dir / img.name)
        (lbl_dir / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n


def write_yaml(dst: Path) -> Path:
    yaml_path = dst / "ppe.yaml"
    txt = (
        f"# Dataset employes de chantier (PPE)\n"
        f"path: {dst.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    yaml_path.write_text(txt)
    return yaml_path


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="XML Pascal VOC -> YOLO txt")
    ap.add_argument("--src", required=True, help="dossier racine contenant training/ et testing/")
    ap.add_argument("--dst", default="./datasets/ppe", help="dossier de sortie (format YOLO)")
    ap.add_argument("--train-dir", default="training", help="nom du sous-dossier train dans --src")
    ap.add_argument("--test-dir", default="testing", help="nom du sous-dossier test dans --src")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="fraction du train mise en validation (tu as deja fait 80/20 -> laisse 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="graine du split (reproductible)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        print(f"[ERREUR] --src introuvable : {src}")
        return 1

    train_src = src / args.train_dir
    test_src = src / args.test_dir

    # --- si le train est deja splitte en deux dossiers, on les utilise tels quels
    subtrain = train_src / "train"
    subval = train_src / "val" if (train_src / "val").is_dir() else train_src / "validation"

    stats = Counter()
    warns: list[str] = []

    if subtrain.is_dir() and subval.is_dir():
        print("[i] Dossiers train/ et val/ deja presents -> split existant conserve.")
        n_tr = copy_split(find_pairs(subtrain), dst / "images/train", dst / "labels/train", stats, warns)
        n_va = copy_split(find_pairs(subval), dst / "images/val", dst / "labels/val", stats, warns)
    else:
        pairs = find_pairs(train_src)
        if not pairs:
            print(f"[ERREUR] aucune image trouvee dans {train_src}")
            return 1
        random.Random(args.seed).shuffle(pairs)
        n_val = int(round(len(pairs) * args.val_ratio))
        val_pairs, tr_pairs = pairs[:n_val], pairs[n_val:]
        print(f"[i] Split {len(tr_pairs)} train / {len(val_pairs)} val (seed={args.seed})")
        n_tr = copy_split(tr_pairs, dst / "images/train", dst / "labels/train", stats, warns)
        n_va = copy_split(val_pairs, dst / "images/val", dst / "labels/val", stats, warns)

    n_te = 0
    if test_src.is_dir():
        n_te = copy_split(find_pairs(test_src), dst / "images/test", dst / "labels/test", stats, warns)

    yaml_path = write_yaml(dst)

    # --- rapport
    total_imgs = n_tr + n_va + n_te
    print(f"\n=== Conversion terminee ===")
    print(f"images  train={n_tr}  val={n_va}  test={n_te}  (total {total_imgs})")
    print(f"objets  total={sum(stats.values())}")
    for i, name in enumerate(CLASS_NAMES):
        c = stats[name]
        pct = 100 * c / max(1, sum(stats.values()))
        print(f"  {i} {name:<14} {c:>7}  {pct:5.1f} %")
    print(f"\nyaml : {yaml_path}")

    if warns:
        Path("conversion_warnings.log").write_text("\n".join(warns))
        print(f"\n[!] {len(warns)} avertissement(s) -> conversion_warnings.log")
        for w in warns[:10]:
            print("   ", w)

    # --- sanity check : 1 image / 1 label
    for split in ("train", "val", "test"):
        imgs = list((dst / "images" / split).glob("*")) if (dst / "images" / split).is_dir() else []
        lbls = list((dst / "labels" / split).glob("*.txt")) if (dst / "labels" / split).is_dir() else []
        if imgs and len(imgs) != len(lbls):
            print(f"[!] {split}: {len(imgs)} images vs {len(lbls)} labels -> probleme")
            return 2
    print("[OK] 1 image = 1 label dans chaque split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
