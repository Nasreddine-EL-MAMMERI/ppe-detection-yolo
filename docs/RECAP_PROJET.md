# Récapitulatif du projet — Détection EPI chantier (YOLOv8)

*Dernière mise à jour : 17/11/2026*

---

## 1. Où en est le projet

| # | Étape | État |
|---|---|---|
| 1 | Environnement (venv Windows, CPU, torch 2.15, ultralytics 8.4) | ✅ fait |
| 2 | Dataset en place, structure vérifiée | ✅ fait |
| 3 | Conversion XML → YOLO + split 80/20 | ✅ fait |
| 4 | Analyse du dataset (déséquilibre, taille des objets) | ✅ fait |
| 5 | Vérification visuelle des annotations | ✅ fait |
| 6 | **Entraînement GPU (Kaggle T4, 100 epochs)** | ⏳ **à faire** |
| 7 | Lecture des courbes + matrice de confusion | ⏳ à faire |
| 8 | Évaluation finale sur le test set | ⏳ à faire |
| 9 | Inférence / export ONNX | ⏳ à faire |

---

## 2. Configuration finale retenue

```
C:\Users\elhad\ppe_project\          ← projet unique, tout au même endroit
├── .venv\                           ← Python, torch 2.15+cpu, ultralytics 8.4
├── scripts\                         ← pipeline local
│   ├── check_env.py
│   ├── xml_to_yolo.py
│   ├── dataset_stats.py
│   ├── train.py                     ← presets --cpu / --gpu, --fraction
│   └── evaluate.py
├── verif.py                         ← contrôle visuel des annotations
├── configs\yolov8n-ppe.yaml         ← modèle custom ~1.3 M params (CPU only)
├── notebooks\ppe_kaggle.ipynb       ← pipeline Kaggle (lecture seule gérée)
├── data\dataset_brut\               ← training\ (1080) + testing\ (250)
├── datasets\ppe\                    ← généré : images/, labels/, ppe.yaml
└── runs\                            ← généré à l'entraînement
```

Machine locale : 8,4 Go RAM (1,2 Go libres), 2 threads CPU, pas de CUDA → **entraînement réel sur Kaggle uniquement**.

---

## 3. Le dataset en chiffres

| | images | objets | obj/img |
|---|---|---|---|
| train | 864 | 5 935 | 6,87 |
| val | 216 | 1 625 | 7,52 |
| test | 250 | 1 649 | 6,60 |
| **total** | **1 330** | **9 209** | 6,92 |

Distribution des classes (train) :

| classe | instances | part | taille médiane à 640 px |
|---|---|---|---|
| person | 2 505 | 42,2 % | ~162 px |
| vest | 1 142 | 19,2 % | ~101 px |
| helmet_yellow | 837 | 14,1 % | ~47 px |
| helmet_white | 786 | 13,2 % | ~41 px |
| helmet_blue | 330 | 5,6 % | ~43 px |
| helmet_red | 335 | 5,6 % | ~37 px |

Conversion : 0 avertissement, 1 image = 1 label partout. Split reproductible (seed 42).
Distributions train/val/test quasi identiques → split sain.

---

## 4. Les 5 décisions à savoir défendre

1. **`imgsz=640` obligatoire.** Les casques font 37–47 px. À 416 ils tombent à 24–31 px, à 320 à 18–24 px — sous la limite de détection de la stride 8. Le 320 local ne sert qu'au test de fumée.
2. **`yolov8n.pt` pré-entraîné, pas from-scratch.** 864 images seulement + `person` existe dans COCO → transfert de features, +10 à +20 mAP attendus. Le custom `yolov8n-ppe.yaml` (1,3 M params) est réservé au CPU.
3. **Test set intouché jusqu'à la fin.** Valider sur val, rapporter sur test. Une seule évaluation finale.
4. **`conf=0.001` en évaluation**, pas 0,25 (sinon −5 points de mAP artificiels).
5. **Lire la mAP par classe, jamais la globale.** Avec un déséquilibre de 7,5× (person vs helmet_blue), une mAP50 globale de 0,75 peut cacher person=0,90 et helmet_blue=0,55.

---

## 5. Défauts connus du dataset (constatés, acceptés)

| Défaut | Impact | Décision |
|---|---|---|
| Boîtes `vest` trop grandes (descendent aux genoux, incluent les bras) | mAP50-95 bridée, confusion person↔vest | **accepté** : l'erreur est *systématique*, donc cohérente. Ré-annoter 1330 images n'en vaut pas le coût |
| Quelques boîtes `person` qui se chevauchent | doublons possibles | à revérifier si `person` sort sous 0,85 |
| Casques petits (37–47 px) | casques colorés 15–20 points sous `person` | **contrainte de données**, pas un bug |

Trois défauts repérés lors du contrôle visuel, un seul (gilets) mériterait une correction — reportée volontairement.

---

## 6. Prochaine action : Kaggle

**Étape A — sur le PC**
```powershell
cd C:\Users\elhad\ppe_project\data
Compress-Archive -Path .\dataset_brut\* -DestinationPath C:\Users\elhad\ppe_upload.zip
```

**Étape B — Kaggle**
1. *Datasets* → **New Dataset** → upload `ppe_upload.zip` → nom `ppe-raw` → *Create*
2. *Code* → **New Notebook** → `File → Import Notebook` → `ppe_kaggle.ipynb`
3. *Settings* : `Accelerator = GPU T4 x2`, `Internet = On`
4. **Add Data** → ton dataset `ppe-raw`
5. Exécuter les cellules dans l'ordre

**Étape C — pendant l'entraînement (~45–70 min)**

| À surveiller | Signe sain | Alerte |
|---|---|---|
| `box_loss` / `cls_loss` | descendent régulièrement | stagnation dès l'epoch 5 |
| `metrics/mAP50(B)` | monte puis se stabilise | reste sous 0,3 |
| `val_batch*_pred.jpg` | boîtes plausibles dès l'epoch 20 | rien de détecté |
| RAM/VRAM | stable | OOM → `batch=8` |

**Résultats attendus** : mAP50 globale **0,70–0,85**, `person` 0,88–0,93, `vest` ~0,75–0,85, casques 0,55–0,75. C'est le plafond réaliste pour 864 images et des objets de 40 px.

---

## 7. Ce qui suit après le premier run

| Observation | Action |
|---|---|
| `helmet_blue` / `helmet_red` < 0,40 | `copy_paste=0.3`, +50 epochs, ou `yolov8s.pt` |
| Confusion `white` ↔ `yellow` forte | vérifier les annotations (matrice de confusion) |
| Tout plafonne vers 0,70 avec casques faibles | passer à `imgsz=960` |
| Sur-apprentissage (val plate, train ↑) | `dropout=0.1`, moins d'epochs |

Puis : évaluation unique sur `test`, export ONNX pour l'inférence CPU, et éventuellement la règle métier « personne sans casque = violation EPI ».

---

## 8. Commandes de référence

```powershell
# contrôle visuel des annotations (5 s)
python verif.py --data .\datasets\ppe --split train --n 12

# test fumée local
python scripts\train.py --cpu --epochs 2 --imgsz 320 --batch 2 --workers 0 --fraction 0.1 --name ppe_smoke

# évaluation finale (une seule fois)
python scripts\evaluate.py --weights runs\detect\ppe_v1\weights\best.pt --split test --visualize

# inférence
python -c "from ultralytics import YOLO; m=YOLO('best.pt'); m.predict('photo.jpg', conf=0.4, save=True)"
```

---

## 9. Pour le CV

> Projet personnel (en cours) — Détection d'EPI chantier, YOLOv8/PyTorch
> Pipeline data complet (1330 annotations XML→YOLO, contrôle qualité, split
> reproductible), analyse du déséquilibre de classes, entraînement GPU (Kaggle).

À compléter avec la mAP réelle **dès que le run Kaggle est terminé** — c'est la seule
chose qui manque pour que le projet passe de « en cours » à « terminé ».
