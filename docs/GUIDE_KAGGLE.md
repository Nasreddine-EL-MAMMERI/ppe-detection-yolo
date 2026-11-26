# Guide Kaggle — étape par étape

Tu as ton dataset **déjà converti** dans
`C:\Users\elhad\ppe_project\datasets\ppe` (images/train, images/val, images/test,
labels/…, ppe.yaml). Voici exactement quoi faire, dans l'ordre.

Notebook à utiliser : **`notebooks/ppe_kaggle_entrainement.ipynb`**
(version adaptée : le dataset est déjà en YOLO, il n'y a plus de conversion à faire).

---

# ÉTAPE 1 — Zipper le dataset

Dans **PowerShell** :

```powershell
cd C:\Users\elhad\ppe_project\datasets
Compress-Archive -Path .\ppe\* -DestinationPath C:\Users\elhad\ppe_dataset.zip
Get-Item C:\Users\elhad\ppe_dataset.zip | Select-Object Name, Length
```

`Length` doit être autour de **250–400 Mo**. Un zip de 5 Mo = problème (tu as zippé le
mauvais dossier).

Vérifie ce que contient le zip (il doit y avoir `images/`, `labels/`, `ppe.yaml` **à la racine**) :

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::OpenRead("C:\Users\elhad\ppe_dataset.zip").Entries |
    Select-Object -First 10 FullName
```

Attendu : `images/…`, `labels/…`, `ppe.yaml` — **pas** `ppe/images/…`.

> Si tu vois `ppe/images/...`, refais le zip **depuis l'intérieur** du dossier :
> ```powershell
> cd C:\Users\elhad\ppe_project\datasets\ppe
> Compress-Archive -Path * -DestinationPath C:\Users\elhad\ppe_dataset.zip -Force
> ```

**Dis-moi « ok » quand le zip existe et fait ~300 Mo.**

---

# ÉTAPE 2 — Uploader sur Kaggle (comme Dataset)

1. Va sur **kaggle.com** (connecté).
2. Menu de gauche → **Datasets** → bouton **New Dataset** (en haut à droite).
3. Glisse `C:\Users\elhad\ppe_dataset.zip` dans la zone d'upload
   (*Drag & drop your files here*).
4. **Titre du dataset** : `ppe-dataset` (le nom devient le chemin `/kaggle/input/ppe-dataset/`).
5. `Visibility` → **Private** (ton dataset n'a pas à être public).
6. Clique **Create** en bas à droite.
7. Attends la fin du traitement : la page passe de *Processing* à *Ready* (2–5 min).

**Vérification** : sur la page du dataset, tu dois voir le fichier `ppe_dataset.zip`
listé dans l'onglet *Data*.

**Dis-moi « ok » quand le dataset est *Ready*.**

---

# ÉTAPE 3 — Créer le notebook + configurer le GPU

1. Menu de gauche → **Code** → bouton **New Notebook**.
2. Dans le notebook qui s'ouvre : menu **File** → **Import Notebook** →
   upload `ppe_kaggle_entrainement.ipynb` (récupère-le dans le zip du projet).
3. Panneau **Settings** (à droite, icône engrenage ou les 3 points ⋮ → *Settings*) :
   - **Accelerator** → `GPU T4 x2`
   - **Internet** → `On`  ← sans ça, `pip install ultralytics` échoue
   - **Persistence** → `Files` (optionnel mais recommandé)
4. Panneau **Data** (à droite) → **Add Data** → onglet **Your Datasets** →
   cherche `ppe-dataset` → **Add**.

**Vérification rapide** — dans une cellule :

```python
import os
print(os.listdir('/kaggle/input/ppe-dataset'))
```

Tu dois voir `['ppe_dataset.zip']`.

**Dis-moi « ok » quand le GPU est activé et le dataset attaché.**

---

# ÉTAPE 4 — Cellule 1 : environnement

Exécute la première cellule du notebook (bouton ▶ à gauche de la cellule, ou
`Maj+Entrée`).

```python
!pip install -q ultralytics
import torch, ultralytics
print('cuda:', torch.cuda.is_available())
```

Attendu :

```
cuda       : True
gpu        : Tesla T4
```

Si `cuda: False` → l'accélérateur n'est pas activé, retourne à l'Étape 3.
Durée : ~1 min.

**Dis-moi « ok » avec la ligne `gpu :`.**

---

# ÉTAPE 5 — Cellule 2 : trouver le zip

Attendu :

```
Archives trouvees :
  /kaggle/input/ppe-dataset/ppe_dataset.zip   (312 Mo)
ZIP retenu : /kaggle/input/ppe-dataset/ppe_dataset.zip
```

Si `assert` échoue → *Add Data* n'a pas été fait.

**Dis-moi « ok ».**

---

# ÉTAPE 6 — Cellule 3 : dézipper

Attendu :

```
Contenu de /kaggle/working/ppe : ['images', 'labels', 'ppe.yaml']
```

⚠️ Si tu vois `['ppe']` au lieu de `['images', 'labels', 'ppe.yaml']`, tu as zippé le
dossier au lieu de son contenu. Ne continue pas : refais l'Étape 1 et re-uploade
(nouvelle version du dataset). Sinon la cellule 5 écrira un yaml pointant dans le vide.

**Dis-moi « ok » avec la liste obtenue.**

---

# ÉTAPE 7 — Cellule 4 : contrôle des splits

Attendu, exactement :

```
train    864 images     864 labels   OK
val      216 images     216 labels   OK
test     250 images     250 labels   OK
```

C'est le même contrôle que tu as fait en local : on vérifie que le transfert n'a rien perdu.

**Dis-moi « ok » avec les 3 lignes.**

---

# ÉTAPE 8 — Cellule 5 : réécrire le ppe.yaml

Ton `ppe.yaml` local contient `path: C:\Users\elhad\...` → inutilisable sous Linux.
La cellule le régénère avec `path: /kaggle/working/ppe`.

Attendu : le contenu du yaml affiché, avec `path: /kaggle/working/ppe`.

**Dis-moi « ok ».**

---

# ÉTAPE 9 — Cellule 6 : LANCEMENT DE L'ENTRAÎNEMENT

C'est la cellule longue. Clique ▶ puis **laisse tourner**, ne ferme pas l'onglet
(Kaggle coupe les sessions inactives au bout de 20–30 min).

Sortie attendue au début :

```
Ultralytics 8.x  Python-3.x  torch-2.x CUDA:0 (Tesla T4, 15109MiB)
Overriding model.yaml nc=80 with nc=6
...
Starting training for 100 epochs...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/100      3.2G      1.842      3.510      1.402        142        640
```

**Ce que tu surveilles pendant les 10 premières epochs :**

| Indicateur | Sain | Problème |
|---|---|---|
| `box_loss` | descend de ~1.8 vers ~1.0 | reste à 1.9 sans bouger |
| `cls_loss` | descend de ~3.5 vers ~1.5 | bloqué à ~4.1 (= log(60)) → labels non lus |
| `Instances` | ~110–160 | 0 → dataset vide |
| `mAP50` à l'epoch 10 | > 0.25 | 0.00 → arrête et dis-le-moi |

Durée : **45–70 min**. Tu peux fermer l'onglet du navigateur, la session continue ;
mais reconnecte-toi avant 20 min d'inactivité pour éviter la coupure.

**Dis-moi « ok » dès que tu vois la ligne `1/100` — pas besoin d'attendre la fin**,
je te dirai quoi surveiller pendant.

---

# ÉTAPE 10 — Cellule 7 : les courbes

Deux images s'affichent : `results.png` (les courbes) et `confusion_matrix.png`.

Ce qu'il faut y lire :

- `metrics/mAP50(B)` doit **monter puis se stabiliser** (plateau normal).
- `box_loss` et `cls_loss` doivent **descendre** puis se tasser.
- Matrice de confusion : regarde la case `person` → `vest` (mon diagnostic sur les
  gilets trop grands) et `helmet_white` ↔ `helmet_yellow`.

Envoie-moi une capture de `results.png` — c'est là que je peux t'aider le plus.

---

# ÉTAPE 11 — Cellule 8 : évaluation sur le TEST set

Une seule fois. Attendu : un tableau par classe.

```
classe                 P       R    mAP50  mAP50-95
person             0.8xx   0.8xx    0.8xx     0.5xx
vest               0.7xx   ...      ...
helmet_blue        ...
...
GLOBAL
```

Ordre de grandeur attendu : **mAP50 global 0,70–0,85**, `person` 0,88–0,93,
`vest` 0,75–0,85, casques colorés 0,55–0,75.

**Envoie-moi ce tableau.**

---

# ÉTAPE 12 — Cellule 9 : récupérer les résultats

`Save Version` (bouton en haut à droite) → **Quick Save** → attends la fin →
onglet **Output** → télécharge `ppe_results.zip`.

Dedans : `weights/best.pt` (le modèle), `results.png`, `confusion_matrix.png`,
`val_batch*_pred.jpg`.

Range-le dans `C:\Users\elhad\ppe_project\runs\detect\ppe_v1\`, tu en auras besoin
pour l'inférence locale.

---

## En cas de problème

| Erreur | Cause | Solution |
|---|---|---|
| `cuda: False` | accélérateur non activé | Settings → GPU T4 x2 |
| `No matching distribution` au pip | Internet Off | Settings → Internet On |
| `FileNotFoundError: ppe/images/train` | zip mal structuré | refaire l'Étape 1 |
| `CUDA out of memory` | batch trop grand | `batch=8` dans la cellule 6 |
| Session coupée | inactivité 20 min | relancer, `last.pt` est sauvegardé |
