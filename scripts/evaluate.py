"""
Etape 4 - Evaluation sur le TEST set (a ne lancer qu'a la toute fin !).

    python scripts/evaluate.py --weights runs/detect/ppe_run/weights/best.pt

Sort un dossier runs/detect/val_<...>/ avec :
  - confusion_matrix.png, PR_curve, F1_curve
  - predictions visualisees (si --visualize)
Et affiche les metriques par classe (c'est la que tu vois si "helmet_blue" est
confondu avec "helmet_white", etc.).
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/ppe_run/weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets/ppe/ppe.yaml"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="seuil bas pour le calcul mAP (0.25 pour visualiser)")
    ap.add_argument("--visualize", action="store_true", help="exporter des images annotees")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.6,
        plots=True,
        save_json=False,
        project=str(ROOT / "runs/detect"),
        name=f"eval_{args.split}",
        exist_ok=True,
    )

    names = model.names
    print("\n=== Metriques par classe ===")
    print(f"{'classe':<16}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for i, idx in enumerate(metrics.ap_class_index):
        print(
            f"{names[int(idx)]:<16}"
            f"{metrics.box.p[i]:>8.3f}{metrics.box.r[i]:>8.3f}"
            f"{metrics.box.ap50[i]:>9.3f}{metrics.box.ap[i]:>10.3f}"
        )
    print(f"\n{'GLOBAL':<16}{metrics.box.mp:>8.3f}{metrics.box.mr:>8.3f}"
          f"{metrics.box.map50:>9.3f}{metrics.box.map:>10.3f}")

    if args.visualize:
        src = ROOT / "datasets/ppe/images" / args.split
        model.predict(
            source=str(src), imgsz=args.imgsz, conf=0.25, save=True,
            project=str(ROOT / "runs/detect"), name=f"visu_{args.split}", exist_ok=True,
        )
        print(f"images annotees -> runs/detect/visu_{args.split}/")


if __name__ == "__main__":
    main()
