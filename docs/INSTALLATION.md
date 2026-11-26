# Installation dans VS Code — pas à pas (Windows / macOS / Linux)

Tu as **deux dossiers séparés**, c'est volontaire et c'est plus propre :

```
📁 Mes documents/
├── 📁 ultralytics/      ← le repo cloné (code source de la lib, on n'y touche pas)
└── 📁 ppe-yolo/         ← TON projet (données, scripts, runs) ← ce dossier
```

> Pourquoi ne pas tout mettre dans `ultralytics/` ?
> Parce que si tu fais `git pull` dans le repo cloné, tu risques des conflits avec tes
> images, tes poids (`.pt`, plusieurs centaines de Mo) et tes `runs/`. Et sur Colab/Kaggle
> tu n'as de toute façon **pas** le repo cloné : tu fais `pip install ultralytics`.
> Un dossier de projet séparé marche dans les deux cas.

---

## 1. Récupérer le dossier `ppe-yolo`

Télécharge `ppe-yolo.zip` depuis ce chat, puis extrais-le là où tu veux travailler :

- **Windows** : clic droit → *Extraire tout…*
- **macOS** : double-clic
- **Linux** : `unzip ppe-yolo.zip`

Tu obtiens un dossier `ppe-yolo/` avec `scripts/`, `configs/`, `notebooks/`, `README.md`,
`requirements.txt`.

---

## 2. Ouvrir le dossier dans VS Code

`Fichier → Ouvrir le dossier…` → sélectionne **`ppe-yolo`** (pas `ultralytics`, pas son
parent).

Tu dois voir dans l'explorateur (barre latérale gauche) :

```
ppe-yolo
├── configs
│   └── yolov8n-ppe.yaml
├── notebooks
│   └── ppe_colab_kaggle.ipynb
├── scripts
│   ├── check_env.py
│   ├── dataset_stats.py
│   ├── evaluate.py
│   ├── train.py
│   └── xml_to_yolo.py
├── README.md
└── requirements.txt
```

Si tu préfères garder `ultralytics` accessible : `Fichier → Ajouter un dossier à
l'espace de travail…` → ajoute `ultralytics`. Les deux apparaîtront côte à côte, mais
c'est `ppe-yolo` qui est la **racine** du projet.

---

## 3. Créer le venv **dans `ppe-yolo`**

Ouvre le terminal intégré : `Terminal → Nouveau terminal` (ou `Ctrl+ù` / `Cmd+ù`).
Vérifie que tu es bien dans `ppe-yolo` :

```bash
pwd            # macOS/Linux → .../ppe-yolo
cd             # Windows → affiche ...\ppe-yolo
```

Puis :

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# si erreur "exécution de scripts désactivée" :
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Le prompt doit maintenant commencer par `(.venv)`. Ensuite :

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Le venv que tu avais créé dans `ultralytics/` : garde-le pour lire le code source, mais
> utilise celui-ci pour le projet. Ne réinstalle pas torch deux fois si tu veux gagner du
> temps — cet install peut prendre 5–10 min.

**Sélectionner l'interpréteur dans VS Code** (sinon VS Code souligne tout en rouge) :
`Ctrl+Shift+P` → `Python: Select Interpreter` → choisis `./.venv/bin/python` /
`.\.venv\Scripts\python.exe`.

---

## 4. Vérifier que tout marche

```bash
python scripts/check_env.py
```

Sortie attendue :

```
torch       : 2.x.x
cuda dispo  : False          ← normal, pas de GPU NVIDIA
mode        : CPU
ultralytics : 8.3.x
```

Si `ultralytics` manque : `pip install ultralytics`.

---

## 5. Ajouter tes données (sans les mélanger au code)

Crée un dossier `data/` à la racine de `ppe-yolo` et mets-y ton dataset brut :

```
ppe-yolo/
├── data/
│   └── dataset_brut/
│       ├── training/   ← 1080 .jpg + 1080 .xml
│       └── testing/    ← 250 .jpg + 250 .xml
├── scripts/
└── ...
```

Puis lance la conversion :

```bash
python scripts/xml_to_yolo.py --src ./data/dataset_brut --dst ./datasets/ppe
```

Le script **copie** les images, il ne déplace rien : ton dossier `data/` reste intact.

Tu obtiens :

```
ppe-yolo/
├── data/dataset_brut/     ← original, ne plus y toucher
├── datasets/ppe/          ← généré : images/, labels/, ppe.yaml
├── runs/detect/           ← généré plus tard par l'entraînement (poids, courbes)
```

---

## 6. `.gitignore` (important si tu pousses sur GitHub)

```gitignore
.venv/
data/
datasets/
runs/
*.pt
*.onnx
__pycache__/
*.pyc
conversion_warnings.log
```

Raison : `runs/` et les poids dépassent vite 500 Mo, et GitHub refuse les fichiers > 100 Mo.
Sur Kaggle/Colab, tu passeras de toute façon par un upload de zip ou un Dataset.

---

## 7. Ordre d'exécution dans VS Code

| Ordre | Commande | Où |
|---|---|---|
| 1 | `python scripts/check_env.py` | terminal VS Code, venv actif |
| 2 | `python scripts/xml_to_yolo.py --src ./data/dataset_brut --dst ./datasets/ppe` | idem |
| 3 | `python scripts/dataset_stats.py --data datasets/ppe` | idem |
| 4 | `python scripts/train.py --cpu --epochs 3 --imgsz 320 --name ppe_smoke` | idem (~10 min) |
| 5 | Ouvrir `notebooks/ppe_colab_kaggle.ipynb` → uploader sur Colab → 100 epochs GPU | Colab |
| 6 | Télécharger `best.pt` → le poser dans `ppe-yolo/runs/detect/ppe_v1/weights/` | — |
| 7 | `python scripts/evaluate.py --weights runs/detect/ppe_v1/weights/best.pt --split test --visualize` | terminal VS Code |

**Étape 5 en détail** (le seul truc un peu manuel) :

1. Va sur [colab.research.google.com](https://colab.research.google.com) →
   `Fichier → Importer un notebook` → `ppe_colab_kaggle.ipynb`.
2. `Exécution → Modifier le type d'exécution → T4 GPU`.
3. Zippe ton dataset brut, upload-le, dézippe dans `/content/raw`.
4. Uploade aussi les scripts (ou clone ton repo s'il est sur GitHub) pour que
   `python scripts/xml_to_yolo.py` fonctionne — le notebook contient déjà la ligne.
5. Exécute les cellules. À la fin : `files.download(...)` pour récupérer `best.pt` et les
   courbes.

---

## 8. Astuces VS Code spécifiques

- **Exécuter un script** : bouton ▶ en haut à droite (lance avec l'interpréteur sélectionné).
- **Voir les images annotées** : elles apparaissent dans `runs/detect/...`, clic dessus pour
  les prévisualiser (extension `vscode-pdf`/image intégrée).
- **Comparer des `results.png`** de plusieurs runs : extension *Image Preview* ou ouvre deux
  onglets côte à côte.
- **Colab à la place du GPU local** : l'extension *Jupyter* de VS Code permet aussi de se
  connecter à un runtime Colab, mais c'est plus fragile que le navigateur pour un premier
  projet — commence simple.

---

## Récapitulatif de la réponse à ta question

- **Oui**, crée un nouveau dossier de projet (`ppe-yolo`) et ouvre-le dans VS Code —
  ne travaille **pas** dans `ultralytics/`.
- Tu n'as **rien à copier dans le repo Ultralytics**. Tout ce dont tu as besoin est dans le
  zip : `scripts/`, `configs/`, `notebooks/`, `README.md`, `requirements.txt`.
- Ajoute simplement `data/` (tes images) à côté.
- Le venv, lui, doit être **dans `ppe-yolo/`** et rester activé dans le terminal intégré.


---

## ⚠️ Dépannage : dossier `ppe-yolo/ppe-yolo/ppe-yolo`

Si ton terminal affiche `C:\Users\<toi>\ppe-yolo\ppe-yolo\ppe-yolo`, l'extraction a
imbriqué les dossiers (l'extracteur Windows le fait quand il existe déjà un dossier du
même nom). **Il ne faut pas créer le venv dans le dossier le plus profond** : il faut
d'abord remettre l'arborescence à plat.

### 1. Trouver le vrai dossier racine

```powershell
Get-ChildItem -Recurse -Filter requirements.txt -Path C:\Users\elhad\ppe-yolo |
    Select-Object FullName
```

Le dossier qui **contient directement** `requirements.txt` (et `scripts/`, `configs/`,
`notebooks/`, `README.md`) est la vraie racine. C'est le seul endroit où tu dois travailler.

### 2. Aplatir (une seule fois)

```powershell
cd C:\Users\elhad
Rename-Item ppe-yolo ppe-yolo_old
Move-Item ppe-yolo_old\ppe-yolo\ppe-yolo ppe-yolo
Remove-Item -Recurse -Force ppe-yolo_old
```

> Adapte la profondeur si tu as 4 niveaux au lieu de 3. Vérifie ensuite :
> ```powershell
> cd C:\Users\elhad\ppe-yolo
> dir
> ```
> Tu dois voir **`scripts  configs  notebooks  README.md  requirements.txt`** — si tu vois
> encore un dossier `ppe-yolo` seul, c'est qu'il reste un niveau.

### 3. Créer le venv **ici**

```powershell
cd C:\Users\elhad\ppe-yolo        # <-- dossier contenant requirements.txt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Erreur fréquente, la plus courante sur Windows :

```
.\.venv\Scripts\Activate.ps1 : l'exécution de scripts est désactivée sur ce système
```

→ `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` puis relance l'activation.
(Alternative si tu ne veux pas toucher à la policy : `.\.venv\Scripts\activate.bat` dans
un terminal **Invite de commandes** au lieu de PowerShell.)

### 4. Vérifier la version de Python

`python -V` doit donner **3.10 à 3.12** idéalement. Si tu es en **3.13**, c'est supporté
par torch ≥ 2.6, mais si `pip install` échoue sur `torch`, installe Python 3.11 ou 3.12
depuis [python.org](https://www.python.org/downloads/) et recrée le venv avec
`py -3.12 -m venv .venv`.
