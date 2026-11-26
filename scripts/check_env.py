"""
Etape 0 - Verification de l'environnement avant de lancer quoi que ce soit.

    python scripts/check_env.py

Verifie : python, torch, device (cpu/cuda), ultralytics, RAM disque libre.
"""
from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


def main() -> None:
    print("=" * 62)
    print(f"Python      : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"Executable  : {sys.executable}")
    if not Path(sys.prefix).name == "" and "venv" not in sys.prefix and "virtualenv" not in sys.prefix:
        print("  [!] On dirait que tu n'es PAS dans un venv active. Fais : source .venv/bin/activate")
    print("=" * 62)

    try:
        import torch
    except ImportError:
        print("[X] torch absent -> pip install torch torchvision (version CPU)")
        return
    print(f"torch       : {torch.__version__}")
    print(f"cuda dispo  : {torch.cuda.is_available()}")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev == "cuda:0":
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
    else:
        print("mode        : CPU (normal sur ton laptop, pas de carte NVIDIA)")
        print(f"threads CPU : {torch.get_num_threads()}")

    try:
        import ultralytics
        print(f"ultralytics : {ultralytics.__version__}")
    except ImportError:
        print("[X] ultralytics absent -> pip install ultralytics")

    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM         : {mem.total / 1e9:.1f} Go (libre {mem.available / 1e9:.1f} Go)")
    except ImportError:
        pass
    free = shutil.disk_usage(".").free / 1e9
    print(f"Disque libre: {free:.1f} Go")

    print("=" * 62)
    print("Diagnostic rapide :")
    print("  - Sur CPU, entraine avec imgsz=416/512, batch=4..8, workers=2, epochs=60-100.")
    print("  - Pour un vrai entrainement : Colab (T4) ou Kaggle (T4/P100) -> imgsz=640, batch=16.")
    print("  - Garde test/ intact jusqu'a la derniere evaluation.")


if __name__ == "__main__":
    main()
