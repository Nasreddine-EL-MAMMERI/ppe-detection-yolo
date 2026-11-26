"""
Etape 3 - Entrainement.

Trois modes, du plus leger au plus lourd :

  1) CPU / laptop (ton cas)  -> modele custom nano + imgsz 416
     python scripts/train.py --cpu

  2) Colab / Kaggle (T4)     -> yolov8n pre-entraine + imgsz 640
     python scripts/train.py --gpu

  3) Ablation (comparer)     -> --model yolov8n.pt / --model yolov8s.pt ...

Tout est logge automatiquement dans runs/detect/<name>/ (poids, courbes, confusion matrix).
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets/ppe/ppe.yaml"
DEFAULT_CFG = ROOT / "configs/yolov8n-ppe.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default=None,
                    help="yolov8n.pt (pre-entraine) ou .yaml (from scratch). Defaut selon --cpu/--gpu")
    ap.add_argument("--cpu", action="store_true", help="preset laptop sans GPU")
    ap.add_argument("--gpu", action="store_true", help="preset Colab/Kaggle")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--name", default="ppe_run")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # ---- presets -------------------------------------------------------------
    if args.cpu:
        model = args.model or str(DEFAULT_CFG)   # leger, from scratch
        epochs, imgsz, batch, workers = 100, 416, 4, 2
        amp = False
    else:
        model = args.model or "yolov8n.pt"       # pre-entraine = convergence bcp plus rapide
        epochs, imgsz, batch, workers = 100, 640, 16, 8
        amp = True

    # overrides CLI
    epochs = args.epochs or epochs
    imgsz = args.imgsz or imgsz
    batch = args.batch or batch

    print(f"modele   : {model}")
    print(f"epochs={epochs} imgsz={imgsz} batch={batch} workers={workers}")
    print(f"data     : {args.data}\n")

    from ultralytics import YOLO

    yolo = YOLO(model)

    results = yolo.train(
        data=args.data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu" if args.cpu else 0,
        workers=workers,
        amp=amp,
        project=str(ROOT / "runs/detect"),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        # --- optimisation memoire / vitesse CPU
        cache=False,          # True si tu as >16 Go de RAM (met les images en RAM)
        rect=False,           # True accelere un peu mais nuit un peu a l'objectif
        # --- augmentation : pensee pour un dataset de chantier
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, close_mosaic=15, mixup=0.0,
        # --- regularisation
        patience=25,          # early stopping : arret si pas d'amelioration
        cos_lr=True,
        lr0=0.01 if args.cpu else 0.01,
        lrf=0.01,
        optimizer="auto",
        seed=42,
        val=True,
        plots=True,
        save_period=10,       # checkpoint tous les 10 epochs (utile si Colab coupe)
    )

    print("\n=== Fin d'entrainement ===")
    print(f"poids    : runs/detect/{args.name}/weights/best.pt")
    print(f"metriques: runs/detect/{args.name}/results.png, confusion_matrix.png")
    if results is not None:
        print(f"map50    : {results.box.map50:.4f}   map50-95 : {results.box.map:.4f}")


if __name__ == "__main__":
    main()
