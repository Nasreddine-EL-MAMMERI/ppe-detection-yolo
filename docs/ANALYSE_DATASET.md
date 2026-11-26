# Diagnostic de ton dataset + stratégie d'entraînement

## Ce que les chiffres disent (et pourquoi ça change le plan)

### 1. Les casques font 35–50 px à imgsz=640 → `imgsz=640` est **obligatoire**

| classe | taille médiane | px à 640 | px à 416 | px à 320 |
|---|---|---|---|---|
| person | 6.4 % | ~162 | ~105 | ~81 |
| vest | 2.5 % | ~101 | ~66 | ~51 |
| helmet_blue | 0.47 % | ~43 | **~28** | **~22** |
| helmet_red | 0.34 % | ~37 | **~24** | **~18** |
| helmet_white | 0.42 % | ~41 | **~27** | **~21** |
| helmet_yellow | 0.56 % | ~47 | **~31** | **~24** |

YOLOv8 détecte à partir de la stride 8 (P3), soit une cellule de 8 px dans l'image
redimensionnée. À 416, un casque de 24 px ne couvre que 3 cellules : la classe est
techniquement dans les données mais à la limite de la représentation. **À 320, c'est mort.**

Conclusion : l'entraînement réel se fait en **640**. Le 320/416 de ton laptop ne sert qu'à
vérifier que la chaîne fonctionne, jamais à juger la qualité.

### 2. RAM : 8,4 Go dont **1,2 Go libres** → contrainte dure

1,2 Go libres n'est pas un manque de RAM, c'est un signe que d'autres applications
(Chrome, VS Code, Windows) occupent déjà 7 Go. YOLO en 640 sur CPU a besoin de ~2 Go
pour un batch de 4. Trois conséquences :

- Ferme Chrome avant tout entraînement local.
- `workers=0` (pas 2) : chaque worker est un **processus** qui duplique le dataset.
- `batch=2`, `imgsz=448` maximum pour le test local, et `--fraction 0.1`.
- Sur Colab/Kaggle (12–16 Go dédiés), `batch=16, imgsz=640` sans problème.

### 3. Déséquilibre : 42 % / 19 % / 5,6 % / 5,6 % / 13 % / 14 %

`person` fait **7,5×** `helmet_blue`. Ce n'est pas un problème à corriger par la perte
(pas de class weights dans Ultralytics standard) mais à compenser par l'augmentation et à
garder en tête à la lecture des résultats : une mAP50 globale de 0,75 peut cacher
`person`=0,90 et `helmet_blue`=0,55. **Regarde toujours la mAP par classe.**

Bonne nouvelle : `helmet_blue` et `helmet_red` ne sont pas des classes rares au point d'être
inexploitables (330–335 instances en train). C'est juste sous la moyenne. Si elles sortent
sous 0,40 après un premier run, alors seulement on envisage `copy_paste` ou une
ré-annotation.

### 4. 7 objets/image en moyenne

C'est **élevé** pour un dataset de chantier, et c'est une très bonne chose : beaucoup
d'instances par image = le modèle apprend vite les co-occurrences (person + vest + casque).
Mais ça explique aussi la charge mémoire : une image 640 avec 7 objets génère ~200 ancres
positives par couche P3.

### 5. Test set 250 images / 1649 objets

Statistiquement solide. Une mAP50 sur 1649 objets a une incertitude d'environ ±1,5 point :
un écart de 2 points entre deux modèles n'est probablement pas significatif.

---

## Stratégie en 3 entraînements

### Run 0 — Test fumée (laptop, ~10 min)

But : détecter un bug de labels/data, pas mesurer la qualité.

```powershell
cd C:\Users\elhad\ppe_project
python scripts\train.py --cpu --epochs 2 --imgsz 320 --batch 2 --workers 0 --fraction 0.1 --name ppe_smoke
```

Si `box_loss` et `cls_loss` **descendent** entre epoch 1 et 2 → la data est saine, tu peux
passer au GPU. Si `cls_loss` reste à ~4.1 (= `log(60)`) sans bouger, tes labels ne sont
pas lus : reviens à l'étape conversion.

### Run 1 — Baseline GPU (Colab/Kaggle, ~40 min)

```python
model = YOLO('yolov8n.pt')
model.train(data='.../ppe.yaml', epochs=100, imgsz=640, batch=16,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
            degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5,
            mosaic=1.0, close_mosaic=15, cos_lr=True, patience=25, seed=42)
```

C'est ta référence. Note la mAP par classe sur la validation.

### Run 2 — Décision selon les résultats

| Observation sur Run 1 | Action |
|---|---|
| `helmet_blue`/`red` < 0.40 | `copy_paste=0.3`, +50 epochs, ou `yolov8s.pt` |
| `person` > 0.90 et casques ~0.60 | Normal. Le plafond est la taille des objets → passer à `imgsz=960` |
| Confusion `helmet_white` ↔ `helmet_yellow` forte | Vérifier les annotations (la matrice de confusion le montre) |
| mAP val stable mais mAP train monte | Sur-apprentissage → `dropout=0.1`, moins d'epochs |
| Faux positifs sur le décor | Ajouter des images négatives (sans annotation) |

---

## Optimisations spécifiques à ton setup

### Windows + CPU : `workers=0`

Sur Windows, chaque worker relance l'import complet (torch, ultralytics) — démarrage de
30 s par worker et duplication mémoire. Avec 1,2 Go libres, `workers=0` est le seul choix
sûr en local.

### `--fraction` pour itérer vite

Le paramètre existe maintenant dans `train.py` :
`--fraction 0.1` utilise ~86 images. Couplé à `--epochs 2`, tu as un cycle de test de
2 minutes au lieu de 40.

### Où mettre `runs/`

Sur le SSD (dans `ppe_project\runs`), jamais sur la clé USB : YOLO écrit des checkpoints
de ~6 Mo à chaque epoch (`last.pt` + `best.pt`), et une écriture USB concurrente du
DataLoader ralentit tout.

---

## Ce qu'il faut graver avant de lancer

1. **Ne touche pas à `datasets/ppe/images/test`** avant la toute fin.
2. **Juge toujours la mAP par classe**, jamais seulement la globale.
3. **Le seuil `conf` en évaluation = 0.001**, pas 0.25 (sinon tu sous-estimes la mAP de
   ~5 points). `evaluate.py` le fait déjà.
4. Un résultat mAP50 de 0.70–0.85 est **normal** pour 1080 images avec des objets de
   40 px. Ne cherche pas 0.95.
