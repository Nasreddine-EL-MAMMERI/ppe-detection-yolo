r"""
Regenere toutes les figures du projet SANS reentrainer.

    python scripts/make_figures.py --weights runs\detect\ppe_v1\weights\best.pt --data .\datasets\ppe

Produit dans ./figures/ :
  - courbes d'entrainement (depuis results.csv si present, sinon ignore)
  - matrice de confusion, courbes PR / F1 / P / R (depuis best.pt)
  - exemples de detections annotees sur le test set
  - un tableau recapitulatif des metriques par classe (CSV + PNG)

Tout est regenerable depuis best.pt SAUF les courbes d'entrainement, qui ont besoin de
results.csv (fichier texte de quelques Ko produit a chaque epoch). Si tu ne l'as plus,
les courbes d'entrainement sont perdues -> il faudrait reentrainer, mais tout le reste
est reconstituable.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

NAMES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def plot_training_curves(csv_path: Path, out_path: Path) -> bool:
    """Reproduit les courbes d'entrainement depuis results.csv."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib absent")
        return False

    if not csv_path.is_file():
        print(f"  [!] {csv_path.name} introuvable -> courbes d'entrainement impossibles")
        return False

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return False

    def col(name: str) -> list[float]:
        vals = []
        for r in rows:
            key = next((k for k in r if k.strip() == name), None)
            if key is None:
                return []
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        return vals

    epochs = col("epoch")
    if not epochs:
        return False

    series = [
        ("train/box_loss", "val/box_loss", "Box loss"),
        ("train/cls_loss", "val/cls_loss", "Cls loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL loss"),
        ("metrics/precision(B)", "metrics/recall(B)", "Precision / Recall"),
        ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "mAP50 / mAP50-95"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    for ax, (tr, va, title) in zip(axes, series):
        for name, style in ((tr, "-"), (va, "--")):
            y = col(name)
            if y:
                ax.plot(epochs, y, style, label=name.split("/")[1])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # LR
    lr = col("lr/pg0")
    if lr:
        axes[5].plot(epochs, lr, color="tab:green")
        axes[5].set_title("learning rate")
        axes[5].set_xlabel("epoch")
        axes[5].grid(alpha=0.3)
    else:
        axes[5].axis("off")

    fig.suptitle("Courbes d'entraînement (reconstruites depuis results.csv)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_class_table(metrics, names, out_png: Path, out_csv: Path) -> None:
    """Tableau des metriques par classe : PNG (pour le README) + CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = []
    for i, idx in enumerate(metrics.ap_class_index):
        rows.append({
            "classe": names[int(idx)],
            "precision": round(float(metrics.box.p[i]), 3),
            "rappel": round(float(metrics.box.r[i]), 3),
            "mAP50": round(float(metrics.box.ap50[i]), 3),
            "mAP50-95": round(float(metrics.box.ap[i]), 3),
        })

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["classe", "precision", "rappel", "mAP50", "mAP50-95"])
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.6))
    ax.axis("off")
    cell = [[r["classe"], f"{r['precision']:.3f}", f"{r['rappel']:.3f}",
             f"{r['mAP50']:.3f}", f"{r['mAP50-95']:.3f}"] for r in rows]
    cell.append(["GLOBAL", f"{metrics.box.mp:.3f}", f"{metrics.box.mr:.3f}",
                 f"{metrics.box.map50:.3f}", f"{metrics.box.map:.3f}"])
    tab = ax.table(cellText=cell,
                   colLabels=["classe", "P", "R", "mAP50", "mAP50-95"],
                   loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1, 1.6)
    for j in range(5):
        tab[0, j].set_facecolor("#2c3e50")
        tab[0, j].set_text_props(color="white", weight="bold")
        tab[len(cell), j].set_facecolor("#ecf0f1")
        tab[len(cell), j].set_text_props(weight="bold")
    ax.set_title("Métriques par classe — set de test", fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=r".\datasets\ppe")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--results-csv", default=None,
                    help="chemin de results.csv (defaut : a cote de best.pt)")
    ap.add_argument("--n-images", type=int, default=12,
                    help="nombre d'images annotees a exporter pour la demo")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights).resolve()

    # --- 1. courbes d'entrainement
    csv_path = Path(args.results_csv) if args.results_csv else (weights.parents[1] / "results.csv")
    print(f"[1/3] Courbes d'entrainement depuis {csv_path}")
    if plot_training_curves(csv_path, out / "courbes_entrainement.png"):
        print(f"      -> {out / 'courbes_entrainement.png'}")
    else:
        print("      -> impossible (results.csv absent). Ce fichier fait ~50 Ko : cherche-le")
        print("         dans ton zip de resultats Kaggle (runs/ppe_v1/results.csv).")

    # --- 2. metriques + figures de validation
    print(f"\n[2/3] Evaluation sur {args.split} + figures")
    model = YOLO(str(weights))
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                  conf=0.001, iou=0.6, plots=True, verbose=False,
                  project=str(out), name="val", exist_ok=True)

    for src, dst in (("confusion_matrix.png", "matrice_confusion.png"),
                     ("confusion_matrix_normalized.png", "matrice_confusion_norm.png"),
                     ("PR_curve.png", "courbe_PR.png"),
                     ("F1_curve.png", "courbe_F1.png"),
                     ("P_curve.png", "courbe_P.png"),
                     ("R_curve.png", "courbe_R.png")):
        p = out / "val" / src
        if p.is_file():
            import shutil
            shutil.copy2(p, out / dst)
            print(f"      -> {out / dst}")
    for p in sorted((out / "val").glob("val_batch*_pred.jpg")):
        import shutil
        shutil.copy2(p, out / p.name)
        print(f"      -> {out / p.name}")

    plot_class_table(m, model.names, out / "metriques_par_classe.png",
                     out / "metriques_par_classe.csv")
    print(f"      -> {out / 'metriques_par_classe.png'} + .csv")

    # --- 3. images annotees pour la demo
    print(f"\n[3/3] {args.n_images} images annotees")
    img_dir = Path(args.data) / "images" / args.split
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)[: args.n_images]
    if imgs:
        model.predict([str(p) for p in imgs], imgsz=args.imgsz, conf=0.35,
                      save=True, project=str(out), name="exemples", exist_ok=True,
                      verbose=False)
        print(f"      -> {out / 'exemples'}")

    print(f"\n=== Figures generees dans {out.resolve()} ===")
    print("A pousser sur GitHub : elles font le README. Verifie qu'aucune image source")
    print("n'est visible en clair si tes donnees sont sous licence (les exemples annotes")
    print("contiennent les photos : ne les pousse que si tu en as le droit).")


if __name__ == "__main__":
    main()
