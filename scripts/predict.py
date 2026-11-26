r"""
Inference locale + regle metier EPI, avec SEUILS PAR CLASSE.

    python predict.py --weights best.pt --source photo.jpg
    python predict.py --weights best.pt --source .\images --rule --save

Pourquoi deux seuils : les erreurs n'ont pas le meme cout en securite.
  - rater une PERSONNE  -> un ouvrier n'est jamais controle  => seuil bas (0.25)
  - inventer un CASQUE  -> une tete nue declaree protegee    => seuil haut (0.45)

Valeurs par defaut issues de la calibration sur 1625 objets :
    person  conf 0.25 -> P=0.841  R=0.872
    casques conf 0.45 -> P=0.903  R=0.834
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAMES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]
HELMETS = {"helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {
    "person": (230, 60, 60),
    "vest": (60, 200, 60),
    "helmet_blue": (60, 130, 255),
    "helmet_red": (230, 60, 230),
    "helmet_white": (245, 245, 245),
    "helmet_yellow": (240, 220, 50),
}


def threshold_for(name: str, conf_person: float, conf_helmet: float, conf_other: float) -> float:
    """Seuil applicable a une classe donnee."""
    if name == "person":
        return conf_person
    if name in HELMETS:
        return conf_helmet
    return conf_other


def filter_boxes(result, conf_person: float, conf_helmet: float, conf_other: float,
                 dedup_iou: float = 0.6, min_height: int = 0,
                 min_width: int = 0, min_area: int = 0):
    """Retourne (xyxy, cls_ids, confs, names) apres filtrage par classe."""
    if result.boxes is None or len(result.boxes) == 0:
        z = np.zeros((0, 4), dtype=np.float32)
        return z, np.zeros((0,), dtype=int), np.zeros((0,), dtype=np.float32), []

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    cf = result.boxes.conf.cpu().numpy()

    keep = np.array([
        cf[i] >= threshold_for(NAMES[cls[i]], conf_person, conf_helmet, conf_other)
        for i in range(len(cls))
    ], dtype=bool)

    xyxy, cls, cf = xyxy[keep], cls[keep], cf[keep]

    # une "personne" trop petite n'est pas une personne : on filtre hauteur ET aire
    if (min_height > 0 or min_width > 0 or min_area > 0) and len(xyxy):
        def plausible(i) -> bool:
            if NAMES[cls[i]] != "person":
                return True
            w = xyxy[i, 2] - xyxy[i, 0]
            h = xyxy[i, 3] - xyxy[i, 1]
            return h >= min_height and w >= min_width and w * h >= min_area
        ok = np.array([plausible(i) for i in range(len(cls))], dtype=bool)
        xyxy, cls, cf = xyxy[ok], cls[ok], cf[ok]

    # doublons
    if dedup_iou < 1.0:
        xyxy, cls, cf = deduplicate(xyxy, cls, cf, dedup_iou)

    names = [NAMES[c] for c in cls]
    return xyxy, cls, cf, names


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2]-a[0])*(a[3]-a[1])) + ((b[2]-b[0])*(b[3]-b[1])) - inter
    return inter / max(1e-9, ua)


def deduplicate(xyxy, cls, cf, thr: float = 0.6, contain: float = 0.8):
    """Supprime les detections redondantes d'une meme classe.

    Deux cas sont traites :
      - boites qui se recouvrent (IoU > thr)
      - boite IMBRIQUEE dans une autre (inclusion > contain) : le cas des personnes
        detectees deux fois, une boite 'torse' dans une boite 'corps entier'.
        On garde alors la plus GRANDE, plus utile pour la regle EPI.
    """
    if len(xyxy) == 0:
        z = np.zeros((0, 4), dtype=np.float32)
        return z, np.zeros((0,), dtype=int), np.zeros((0,), dtype=np.float32)

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep: list[int] = []
    # on parcourt du plus grand au plus petit : une petite boite incluse sera ecartee
    for i in np.argsort(-areas):
        drop = False
        for j in keep:
            if cls[i] != cls[j]:
                continue
            if iou(xyxy[i], xyxy[j]) > thr:
                drop = True
                break
            if containment(xyxy[i], xyxy[j]) > contain:
                drop = True   # i est incluse dans j (plus grande, deja retenue)
                break
        if not drop:
            keep.append(int(i))
    return xyxy[keep], cls[keep], cf[keep]


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction de 'inner' contenue dans 'outer' (0-1)."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (iw * ih) / area


def analyse_violations(xyxy, names, confs) -> list[str]:
    """Toute personne dont le haut du corps ne contient aucun casque = violation."""
    persons = [(xyxy[i], confs[i]) for i, n in enumerate(names) if n == "person"]
    helmets = [xyxy[i] for i, n in enumerate(names) if n in HELMETS]

    out = []
    for p_box, p_conf in persons:
        top = p_box.copy()
        top[3] = p_box[1] + 0.55 * (p_box[3] - p_box[1])
        has_helmet = any(containment(h, top) > 0.4 for h in helmets)
        if not has_helmet:
            out.append(
                f"  PERSONNE SANS CASQUE  conf={p_conf:.2f}  "
                f"boite=[{int(p_box[0])},{int(p_box[1])},{int(p_box[2])},{int(p_box[3])}]"
            )
    return out


def draw(path: Path, xyxy, names, out_path: Path) -> bool:
    """Dessine les boites filtrees (remplace save=True qui n'applique pas nos seuils)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return False
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", max(12, im.size[0] // 55))
    except OSError:
        f = ImageFont.load_default()
    for box, name in zip(xyxy, names):
        col = COLORS.get(name, (255, 255, 255))
        d.rectangle([box[0], box[1], box[2], box[3]], outline=col, width=3)
        d.text((box[0] + 3, max(0, box[1] - 14)), name, fill=col, font=f)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--source", required=True, help="image, dossier, ou 0 pour webcam")
    ap.add_argument("--conf-person", type=float, default=0.25,
                    help="seuil pour 'person' (bas = ne rater personne)")
    ap.add_argument("--conf-helmet", type=float, default=0.45,
                    help="seuil pour les casques (haut = ne pas inventer de casque)")
    ap.add_argument("--conf-other", type=float, default=0.40, help="seuil pour 'vest'")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-height", type=int, default=40,
                    help="hauteur mini d'une boite 'person' en px (0 = desactive)")
    ap.add_argument("--min-width", type=int, default=18,
                    help="largeur mini d'une boite 'person' en px")
    ap.add_argument("--min-area", type=int, default=2500,
                    help="aire mini d'une boite 'person' en px2")
    ap.add_argument("--min-box-iou", type=float, default=0.6,
                    help="seuil de deduplication (1.0 = desactive)")
    ap.add_argument("--rule", action="store_true", help="signaler les personnes sans casque")
    ap.add_argument("--save", action="store_true", help="ecrire les images annotees")
    ap.add_argument("--out", default="runs/detect/predict", help="dossier de sortie")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    src = Path(args.source) if args.source != "0" else None
    files: list[Path] = []
    if src is not None:
        if src.is_dir():
            files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        elif src.is_file():
            files = [src]
        else:
            print(f"[X] source introuvable : {src}")
            return
        if not files:
            print(f"[X] aucune image dans {src}")
            return
        print(f"{len(files)} image(s)\n")

    # seuil le plus bas de tous -> on ne perd aucune boite avant filtrage maison
    conf_min = min(args.conf_person, args.conf_helmet, args.conf_other)
    results = model.predict(
        source=args.source if src is not None else "0",
        imgsz=args.imgsz,
        conf=conf_min,
        verbose=False,
    )

    out_dir = Path(args.out)
    all_violations = []

    for r in results:
        f_path = Path(getattr(r, "path", ""))
        xyxy, cls, cf, names = filter_boxes(
            r, args.conf_person, args.conf_helmet, args.conf_other,
            args.min_box_iou, args.min_height, args.min_width, args.min_area)

        counts: dict[str, int] = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-"
        print(f"{f_path.name:<28} {len(names):>3} objets   {summary}")

        if args.rule:
            v = analyse_violations(xyxy, names, cf)
            if v:
                print(f"  [!] {len(v)} violation(s) EPI")
                for line in v:
                    print(line)
                all_violations.append(f"[{f_path.name}]" + "\n" + "\n".join(v))

        if args.save and f_path.is_file():
            draw(f_path, xyxy, names, out_dir / f_path.name)

    if args.rule:
        rep = Path(args.reports)
        rep.mkdir(parents=True, exist_ok=True)
        out = rep / "violations_epi.txt"
        out.write_text("\n\n".join(all_violations) if all_violations else "Aucune violation detectee.\n")
        print(f"\nRapport -> {out}")

    if args.save:
        print(f"Images annotees -> {out_dir}")
    print(f"\nSeuils : person={args.conf_person}  casques={args.conf_helmet}  vest={args.conf_other}")
    print(f"Filtres : min_height={args.min_height}px  min_width={args.min_width}px  "
          f"min_area={args.min_area}px2  dedup_iou={args.min_box_iou}")


if __name__ == "__main__":
    main()
