r"""
Evalue la REGLE METIER (personne sans casque) contre les annotations de verite.

    python scripts/evaluate_rule.py --weights best.pt --data .\datasets\ppe --split test

Pourquoi : predict.py te dit "voici les violations". Ce script te dit "voici combien
sont vraies, combien sont fausses". C'est la difference entre une demo et un outil.

Principe : les annotations contiennent deja les personnes ET les casques. On peut donc
savoir, pour chaque personne annotee, si elle porte un casque ou non. C'est la verite
terrain de la regle metier.

Metriques produites :
  - nombre de violations reelles (annotations) vs signalees (modele)
  - precision de la regle = violations signalees qui sont vraies
  - rappel de la regle    = violations reelles qui sont signalees
  - detail par cause d'erreur (personne non detectee, casque manque, casque invente)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAMES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]
HELMETS = {"helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
TOP_FRACTION = 0.55
CONTAIN = 0.4
HEAD_MARGIN = 0.15   # on cherche le casque jusqu'a 15 % au-dessus de la boite person


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1])) + ((b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / max(1e-9, ua)


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (iw * ih) / area


def top_part(box: np.ndarray, margin: float = HEAD_MARGIN) -> np.ndarray:
    """Partie haute de la personne, etendue vers le haut pour attraper les casques
    qui depassent d'une boite 'person' detectee sans la tete."""
    t = box.copy()
    h = box[3] - box[1]
    t[1] = box[1] - margin * h          # extension vers le haut
    t[3] = box[1] + TOP_FRACTION * h
    return t


def has_helmet(person: np.ndarray, helmets: list[np.ndarray],
               margin: float = HEAD_MARGIN, contain: float = CONTAIN) -> bool:
    top = top_part(person, margin)
    return any(containment(h, top) > contain for h in helmets)


def load_gt(lbl_path: Path, W: int, H: int):
    persons, helmets = [], []
    if not lbl_path.exists():
        return persons, helmets
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, xc, yc, bw, bh = int(p[0]), *(float(v) for v in p[1:])
        box = np.array([(xc - bw / 2) * W, (yc - bh / 2) * H,
                        (xc + bw / 2) * W, (yc + bh / 2) * H], dtype=np.float32)
        if NAMES[c] == "person":
            persons.append(box)
        elif NAMES[c] in HELMETS:
            helmets.append(box)
    return persons, helmets


def dedup(boxes: np.ndarray, confs: np.ndarray, thr: float = 0.6):
    """Supprime les detections redondantes (meme objet detecte 2 fois)."""
    if len(boxes) == 0:
        return boxes, confs
    order = np.argsort(-confs)
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) <= thr for j in keep):
            keep.append(i)
    return boxes[keep], confs[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=r".\datasets\ppe")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-person", type=float, default=0.25)
    ap.add_argument("--conf-helmet", type=float, default=0.45)
    ap.add_argument("--min-height", type=int, default=40,
                    help="hauteur mini d'une boite 'person' en px (0 = desactive)")
    ap.add_argument("--min-width", type=int, default=18,
                    help="largeur mini d'une boite 'person' en px")
    ap.add_argument("--min-area", type=int, default=2500,
                    help="aire mini d'une boite 'person' en px2")
    ap.add_argument("--min-box-iou", type=float, default=0.6,
                    help="seuil de deduplication des personnes (1.0 = desactive)")
    ap.add_argument("--head-margin", type=float, default=HEAD_MARGIN,
                    help="marge de recherche du casque au-dessus de la boite person (fraction de sa hauteur)")
    ap.add_argument("--contain", type=float, default=CONTAIN,
                    help="fraction mini du casque devant tomber dans la zone haute (0.4 = 40 pourcent)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    root = Path(args.data).resolve()
    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print(f"[X] aucune image dans {img_dir}")
        return

    model = YOLO(args.weights)
    results = model.predict([str(p) for p in imgs], imgsz=args.imgsz,
                            conf=min(args.conf_person, args.conf_helmet), verbose=False)

    tp = fp = fn = 0          # sur les personnes appariees
    fp_phantom = 0            # alertes sur des personnes inexistantes (comptees en FP)
    gt_viol_total = 0
    pred_viol_total = 0
    unmatched_pred = 0        # violations signalees sur des personnes inexistantes
    unmatched_gt = 0          # personnes annotees non detectees
    missed_person_beats_helmet = 0
    tot_gt_pers = 0           # total des personnes annotees (pour le taux de detection)
    detail = []

    for img, r in zip(imgs, results):
        with Image.open(img) as im:
            W, H = im.size
        gt_persons, gt_helmets = load_gt(lbl_dir / f"{img.stem}.txt", W, H)
        tot_gt_pers += len(gt_persons)

        if r.boxes is None or len(r.boxes) == 0:
            p_boxes = np.zeros((0, 4), dtype=np.float32)
            p_confs = np.zeros((0,), dtype=np.float32)
            h_boxes = []
            h_confs = []
        else:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            names = np.array([NAMES[c] for c in cls])
            mP = (names == "person") & (cf >= args.conf_person)
            mH = np.isin(names, list(HELMETS)) & (cf >= args.conf_helmet)
            p_boxes, p_confs = dedup(xyxy[mP], cf[mP], args.min_box_iou)
            if len(p_boxes):
                w = p_boxes[:, 2] - p_boxes[:, 0]
                h = p_boxes[:, 3] - p_boxes[:, 1]
                ok = (h >= args.min_height) & (w >= args.min_width) & (w * h >= args.min_area)
                p_boxes, p_confs = p_boxes[ok], p_confs[ok]
            h_boxes = [b for b in xyxy[mH]]

        # --- verite terrain : violations reelles
        gt_viol_flags = [not has_helmet(p, gt_helmets, args.head_margin, args.contain)
                         for p in gt_persons]
        gt_viol_total += sum(gt_viol_flags)

        # --- predictions : violations signalees
        pred_viol_flags = [not has_helmet(p, h_boxes, args.head_margin, args.contain)
                           for p in p_boxes]
        pred_viol_total += sum(pred_viol_flags)

        # --- appariement personne predite <-> personne annotee
        matched_gt = set()
        for i, pb in enumerate(p_boxes):
            best_j, best_iou = -1, 0.0
            for j, gb in enumerate(gt_persons):
                if j in matched_gt:
                    continue
                v = iou(pb, gb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou < 0.5:
                if pred_viol_flags[i]:
                    unmatched_pred += 1
                    fp_phantom += 1     # une alerte sur un fantome EST une fausse alerte
                continue
            matched_gt.add(best_j)
            if pred_viol_flags[i] and gt_viol_flags[best_j]:
                tp += 1
            elif pred_viol_flags[i] and not gt_viol_flags[best_j]:
                fp += 1
                missed_person_beats_helmet += 1
            elif not pred_viol_flags[i] and gt_viol_flags[best_j]:
                fn += 1

        unmatched_gt += len(gt_persons) - len(matched_gt)

        if args.verbose and (any(gt_viol_flags) or any(pred_viol_flags)):
            detail.append(
                f"{img.name:<20} GT: {len(gt_persons)} pers / {sum(gt_viol_flags)} viol   "
                f"PRED: {len(p_boxes)} pers / {sum(pred_viol_flags)} viol"
            )

    print(f"\n=== Evaluation de la regle 'personne sans casque' sur {args.split} ===")
    print(f"seuils : person={args.conf_person}  casques={args.conf_helmet}  "
          f"min_height={args.min_height}  dedup_iou={args.min_box_iou}\n")

    print(f"violations reelles (annotations) : {gt_viol_total}")
    print(f"violations signalees (modele)    : {pred_viol_total}")
    print()
    print(f"  correctement signalees (TP)    : {tp}")
    print(f"  fausses alertes (FP)           : {fp}   <- personne portant un casque, signalee")
    print(f"  violations ratees (FN)         : {fn}   <- personne sans casque, non signalee")
    print(f"    dont sur personne inexistante (fantome) : {fp_phantom}")
    print(f"  personnes annotees non vues    : {unmatched_gt}")
    print()

    # PRECISION HONNETE : les alertes sur fantomes comptent comme fausses
    prec = tp / max(1, tp + fp + fp_phantom)
    prec_appariees = tp / max(1, tp + fp)
    rec_appariees = tp / max(1, tp + fn)
    # RAPPEL HONNETE : les violations sur personnes jamais detectees comptent comme ratees
    rec = tp / max(1, gt_viol_total)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"  precision de la regle : {prec:.3f}   (fantomes inclus - chiffre honnete)")
    print(f"  precision (appariees) : {prec_appariees:.3f}   (hors fantomes - indicatif)")
    print(f"  rappel (appariees)    : {rec_appariees:.3f}   (hors personnes non detectees)")
    print(f"  rappel HONNETE        : {rec:.3f}   (TP / violations reelles = {tp}/{gt_viol_total})")
    print(f"  F1 HONNETE            : {f1:.3f}")
    print()
    print(f"  plafond structurel : {gt_viol_total - tp - fn} violations portent sur des "
          f"personnes JAMAIS detectees -> inaccessibles a la regle")
    print(f"  -> meme avec une regle parfaite, rappel max = "
          f"{(gt_viol_total - (gt_viol_total - tp - fn)) / max(1, gt_viol_total):.3f}")

    print(f"\nSur {tot_gt_pers} personnes annotees : {unmatched_gt} non detectees "
          f"({100 * unmatched_gt / max(1, tot_gt_pers):.0f} %)")

    print("\nLecture :")
    if fp_phantom > fp:
        print(f"  [!] Cause n1 des fausses alertes : {fp_phantom} personnes FANTOMES "
              f"(vs {fp} casques manques).")
        print("      -> monte --conf-person (0.35 puis 0.45) avant de toucher aux casques.")
    if prec < 0.6:
        print(f"  [!] precision {prec:.2f} : trop de fausses alertes.")
    if rec < 0.6:
        print(f"  [!] rappel {rec:.2f} : trop de violations manquees. Baisse --conf-helmet")
        print("      ou --conf-person (des personnes non detectees = violations invisibles).")
    if prec >= 0.7 and rec >= 0.7:
        print("  Regle exploitable : precision et rappel au-dessus de 0.70.")

    print("\nLe compromis a choisir :")
    print("  - securite stricte : --conf-helmet 0.35  -> plus de rappel, plus de fausses alertes")
    print("  - volume maitrise  : --conf-helmet 0.55  -> moins d'alertes, plus de violations ratees")

    if args.verbose and detail:
        print("\n=== Detail par image (violations GT ou predites) ===")
        for line in detail[:60]:
            print(line)


if __name__ == "__main__":
    main()
