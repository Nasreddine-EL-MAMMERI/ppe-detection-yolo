r"""
Range les figures du zip Kaggle dans docs/figures/ avec les noms attendus par le README.

    python scripts/collect_figures.py --src "C:\Users\elhad\ppe_results\runs\ppe_v1"

Le script :
  1. copie et renomme les figures SANS photo source (surs a publier) ;
  2. liste les fichiers qui CONTIENNENT des photos et refuse de les copier
     (filigranes de banques d'images -> probleme de droits sur un repo public) ;
  3. copie results.csv (indispensable pour regenerer les courbes d'entrainement) ;
  4. affiche les liens Markdown a coller dans le README.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# source Kaggle -> (nom dans le repo, description, sur a publier)
SAFE = {
    "results.png": ("courbes_entrainement.png", "courbes d'apprentissage (que des graphes)"),
    "confusion_matrix.png": ("matrice_confusion.png", "matrice de confusion"),
    "confusion_matrix_normalized.png": ("matrice_confusion_norm.png", "matrice normalisee"),
    "PR_curve.png": ("courbe_PR.png", "courbe precision-rappel"),
    "F1_curve.png": ("courbe_F1.png", "courbe F1 vs confiance"),
    "P_curve.png": ("courbe_P.png", "precision vs confiance"),
    "R_curve.png": ("courbe_R.png", "rappel vs confiance"),
    # Ultralytics prefixe ces courbes par "Box" sur les taches de detection
    "BoxPR_curve.png": ("courbe_PR.png", "courbe precision-rappel"),
    "BoxF1_curve.png": ("courbe_F1.png", "courbe F1 vs confiance"),
    "BoxP_curve.png": ("courbe_P.png", "precision vs confiance"),
    "BoxR_curve.png": ("courbe_R.png", "rappel vs confiance"),
    "labels.jpg": ("distribution_annotations.png", "distribution des annotations"),
    "labels_correlogram.jpg": ("correlogramme.png", "correlogramme des boites"),
}

# contiennent les photos source : NE PAS publier
RISKY_PREFIX = ("val_batch", "train_batch", "predict", "exemples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dossier runs/ppe_v1 extrait du zip Kaggle")
    ap.add_argument("--dst", default="docs/figures")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        print(f"[X] dossier introuvable : {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)

    copied, skipped, risky = [], [], []

    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        name = f.name

        if name in SAFE:
            new_name, desc = SAFE[name]
            shutil.copy2(f, dst / new_name)
            copied.append((new_name, f.stat().st_size, desc))
            continue

        if name.startswith(RISKY_PREFIX):
            risky.append(f)
            continue

        if name == "results.csv":
            shutil.copy2(f, dst / "results.csv")
            copied.append(("results.csv", f.stat().st_size, "courbes regenerables"))
            continue

        if name in ("best.pt", "last.pt"):
            skipped.append((name, f.stat().st_size, "6 Mo, optionnel (GitHub conseille < 1 Go)"))
            continue

    print(f"\n=== Copie vers {dst} ===")
    if copied:
        for n, size, desc in copied:
            print(f"  [OK] {n:<32} {size/1024:>7.0f} Ko   {desc}")
    else:
        print("  [!] aucune figure reconnue")
        print("      Verifie que --src pointe bien sur runs/ppe_v1 (pas sur runs/)")

    if risky:
        print(f"\n=== NON copie : {len(risky)} fichier(s) avec photos source ===")
        for f in risky[:8]:
            print(f"  [--] {f.name}")
        print("  Ces images contiennent les photos du dataset (filigranes de banques d'images).")
        print("  Ne les pousse pas sur un repo public. Options :")
        print("    - les garder en local pour tes demos / entretiens")
        print("    - repo prive si tu veux les montrer a un recruteur")
        print("    - les remplacer par des captures de tes propres photos")

    if skipped:
        print(f"\n=== Ignore : {len(skipped)} fichier(s) ===")
        for n, size, why in skipped:
            print(f"  [--] {n:<12} {size/1e6:.1f} Mo   {why}")

    if copied:
        print("\n=== Markdown a coller dans le README ===")
        for n, _, _ in copied:
            if n.endswith(".png"):
                print(f"![{n[:-4]}](docs/figures/{n})")


if __name__ == "__main__":
    main()
