# Résultats finaux — Détection EPI chantier

## 1. Le modèle (niveau détecteur)

| Métrique | Validation (216 img) | **Test (250 img)** |
|---|---|---|
| mAP50 | 0.891 | **0.881** |
| mAP50-95 | 0.507 | 0.497 |
| Précision | 0.895 | 0.917 |
| Rappel | 0.833 | 0.838 |

Modèle : YOLOv8n pré-entraîné COCO, 3,13 M paramètres, 8,6 GFLOPs.
Entraînement : 100 epochs max, early stopping à l'epoch 63, `imgsz=640`, `batch=16`, T4.
Inférence : 4,7 ms/image sur T4.

### Par classe (test)

| classe | instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| person | 695 | 0.918 | 0.871 | 0.896 | 0.521 |
| vest | 326 | 0.905 | 0.794 | 0.880 | 0.500 |
| helmet_blue | 84 | 0.948 | 0.774 | 0.810 | 0.445 |
| helmet_red | 83 | 0.882 | 0.813 | 0.862 | 0.482 |
| helmet_white | 207 | 0.916 | 0.901 | 0.928 | 0.522 |
| helmet_yellow | 254 | 0.934 | 0.874 | 0.907 | 0.510 |

Écart validation → test : **−1 point de mAP50**. Le modèle généralise, il n'a pas mémorisé.

---

## 2. Le seuil de décision (niveau calibration)

Courbe F1 vs seuil de confiance (1625 objets de validation) :

| conf | P | R | F1 |
|---|---|---|---|
| 0.25 | 0.841 | 0.872 | 0.856 |
| 0.30 | 0.862 | 0.866 | 0.864 |
| 0.45 | 0.903 | 0.834 | **0.867** |
| 0.60 | 0.938 | 0.782 | 0.853 |

**Le F1 est plat entre 0.30 et 0.50** (0.864 → 0.867) : le modèle est bien calibré, le
choix du seuil peut se faire selon le besoin métier sans perte de qualité globale.

Rappel maximal atteignable : **0.903 à conf=0.05**. Les 10 % manquants sont des objets
occultés ou tronqués — aucun réglage de seuil ne les récupère.

---

## 3. Le système (niveau règle métier EPI)

Règle : *une personne dont la zone haute ne contient aucun casque détecté = violation*.
Zone de recherche étendue de 15 % au-dessus de la boîte (le modèle détecte parfois le
corps sans la tête).

### Matrice de décision (test, 93 violations réelles sur 1649 objets)

| `--conf-person` | TP | FP casque | Fantômes | Précision | Rappel honnête |
|---|---|---|---|---|---|
| 0.25 | 50 | 28 | 45 | 0.407 | 0.538 |
| 0.35 | 50 | 27 | 34 | 0.450 | 0.538 |
| 0.45 | 46 | 25 | 25 | 0.479 | 0.495 |
| 0.55 | 43 | 19 | 15 | **0.558** | 0.462 |

*(les runs 0.25–0.45 ont été faits sans la marge de 15 % ; le run 0.55 avec — d'où le
gain de précision supplémentaire)*

### Le plafond structurel

Avec `--conf-person 0.55` :

```
93 violations réelles
 43 correctement signalées         (TP)
  7 ratées sur des personnes vues  (FN)
 43 portent sur des personnes JAMAIS détectées  <- inaccessibles
```

**Rappel maximal théorique du système : 0.538**, même avec une règle parfaite. 116 des
695 personnes annotées (17 %) ne sont pas détectées.

---

## 4. La conclusion du projet

**Le facteur limitant n'est pas les casques, c'est le détecteur de personnes.**

C'est contre-intuitif : `person` a la meilleure mAP50 du dataset (0.896) et le plus grand
nombre d'instances (695). Mais les personnes lointaines, petites ou partiellement
occultées sont soit manquées (116), soit inventées (15 fantômes). Une violation sur une
personne invisible ne peut pas être signalée ; une alerte sur une personne inexistante est
une fausse alerte.

**L'outil est un pré-filtre, pas un système autonome.** À `--conf-person 0.35
--conf-helmet 0.45` : ~50 vraies violations trouvées sur 93, avec une alerte sur deux
qui est fausse. C'est utilisable avec vérification humaine, pas en surveillance
automatique.

---

## 5. Réglage recommandé pour l'usage

```powershell
$py = ".\.venv\Scripts\python.exe"
$p  = (Get-ChildItem .\scripts\pred*.py).FullName
$w  = (Get-ChildItem .\runs -Recurse -Filter best.pt | Select-Object -First 1).FullName

& $py $p --weights $w --source .\datasets\ppe\images\test `
    --conf-person 0.35 --conf-helmet 0.45 --rule --save --out .\predictions
```

Justification : `--conf-person 0.35` maximise le rappel sans générer trop de fantômes ;
`--conf-helmet 0.45` évite d'inventer des casques (l'erreur dangereuse en sécurité).

→ **Confirmation à faire** : relancer `evaluate_rule.py` avec `--conf-person 0.35` pour
mesurer l'effet combiné de ce seuil et de la marge de 15 %. C'est le seul run qui reste.

---

## 6. Pistes d'amélioration, par rentabilité

| Priorité | Action | Coût | Gain attendu |
|---|---|---|---|
| 1 | `imgsz=960` (dataset HR requis) | 1 h 30 GPU | +2 à 4 mAP50, **surtout : moins de personnes lointaines manquées** |
| 2 | `yolov8s` en comparaison | 2 h GPU | +2 à 4 mAP50 |
| 3 | Ré-annoter les boîtes `vest` | semaines | +5 à 10 mAP50-95 |
| 4 | Second détecteur dédié aux casques sur crops | jours | améliore la règle métier |

**Ce qui ne marchera pas** : plus d'epochs (early stopping à 63/100), class weights,
plus d'augmentation (aucun sur-apprentissage), changer l'optimizer.

---

