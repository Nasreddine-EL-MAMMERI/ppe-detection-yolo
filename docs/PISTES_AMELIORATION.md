# Comment améliorer le modèle — plan par ordre de rentabilité

## État des lieux honnête

| Métrique (TEST, 250 images / 1649 objets) | Valeur |
|---|---|
| mAP50 | **0.881** |
| mAP50-95 | 0.497 |
| Précision globale | 0.917 |
| Rappel global | 0.838 |

**Ce n'est pas un mauvais modèle.** mAP50 = 0.88 sur un dataset de 1330 images annotées à la
main, c'est un bon résultat. Le rappel de 0.838 ne veut pas dire « il rate 16 % des objets
utiles » : il est mesuré **au seuil qui maximise le F1**, pas au seuil que tu utiliserais en
production. C'est un point crucial (voir §1).

## Rappel par classe — les vrais chiffres

| classe | instances test | Rappel | Manqués |
|---|---|---|---|
| helmet_white | 207 | 0.901 | 10 % |
| helmet_yellow | 254 | 0.874 | 13 % |
| person | 695 | 0.871 | 13 % |
| helmet_red | 83 | 0.813 | 19 % |
| vest | 326 | 0.794 | 21 % |
| **helmet_blue** | **84** | **0.774** | **23 %** |

Le pire cas est `helmet_blue` à **23 %**, pas 30 %. Et avec seulement 84 instances dans le
test, l'intervalle de confiance est de ±8 points : le vrai rappel de `helmet_blue` est
quelque part entre 0.69 et 0.85. Le 0.704 que tu as vu était le rappel `vest` **sur la
validation**, il n'est pas confirmé sur le test (0.794).

---

## §0 — Nuance essentielle : quel rappel, quelle précision ?

En sécurité, on veut maximiser le rappel — **mais pas sur les mêmes classes.** Ta règle métier
est « une personne sans casque détecté = violation ». Décompose-la :

| Erreur | Conséquence | Métrique à privilégier |
|---|---|---|
| `person` manquée (FN) | un ouvrier n'est **jamais contrôlé** | **rappel de `person`** ⬆ |
| casque réel manqué (FN) | fausse alerte, vérification inutile | rappel des casques (agréable, pas critique) |
| **faux casque détecté (FP)** | une tête nue est déclarée protégée → **danger réel** | **précision des casques** ⬆ |

Autrement dit : ton intuition « en sécurité on préfère le rappel » est **vraie pour
`person`** et **inversée pour `helmet_*`**. Un faux casque masque une violation ; c'est
l'erreur la plus dangereuse du système, et c'est une erreur de précision.

Bonne nouvelle : ta précision est déjà haute (0.882–0.948 par classe). C'est le bon côté de
la courbe pour la règle métier.

---

## §1 — Gain gratuit : calibrer le seuil de confiance (+3 à 6 points de rappel)

**Coût : 2 minutes. Aucun réentraînement.**

L'évaluation Ultralytics rapporte P/R au seuil optimal F1. En production tu choisis le seuil.
Baisser `conf` de 0.40 → 0.25 augmente le rappel de plusieurs points en perdant 1–2 points de
précision. Le bon seuil se lit sur `F1_curve.png`, ou se calcule :

```powershell
cd C:\Users\elhad\ppe_project
python scripts\calibrate.py --weights runs\detect\ppe_v1\weights\best.pt --data .\datasets\ppe --split val --target-recall 0.95
```

Sortie : un tableau `conf / P / R / F1` global, le seuil optimal F1, le seuil pour atteindre
95 % de rappel, et **le seuil par classe** (parce que `helmet_blue` et `person` n'ont pas le
même optimum).

À utiliser ensuite :

```powershell
python scripts\predict.py --weights best.pt --source .\images_test --conf 0.25 --rule
```

**C'est le meilleur retour sur investissement de toute cette liste.**

---

## §2 — Gain quasi gratuit : résolution d'entraînement (imgsz 640 → 960)

**Coût : ~2× le temps d'entraînement (1 h 30 sur T4). Gain attendu : +2 à 4 mAP50, surtout sur
les casques.**

Tes casques font 37–47 px à 640. À 960, ils font 55–70 px : deux fois plus de cellules de
grille, donc des features plus riches. C'est la seule amélioration qui attaque directement la
cause physique du problème.

⚠️ Nécessite les images **haute résolution**. Si tu as uploadé `ppe_light` (800 px max), tu es
limité : il faut le dataset original (`ppe_kaggle.zip`, 418 Mo) ou refaire le `shrink` avec
`--max-side 960`.

```python
model.train(data=YAML, imgsz=960, batch=8, epochs=100, ...)  # batch réduit : plus de pixels
```

---

## §3 — Gain modéré : TTA à l'évaluation (+1 à 2 mAP)

**Coût : 2 minutes, purement à l'inférence. Aucun réentraînement.**

Le test-time augmentation moyenne les prédictions sur l'image et son miroir. Gratuit en
qualité, ×2 en temps d'inférence :

```python
m = best.val(data=YAML, split='test', imgsz=960, augment=True, conf=0.001)
```

`predict.py` ne le fait pas (trop lent en production), mais c'est une façon honnête de gagner
1–2 points si tu publies un chiffre final.

---

## §4 — Gain modéré : modèle plus gros (yolov8n → yolov8s)

**Coût : 2× le temps d'entraînement. Gain attendu : +2 à 4 mAP50. Risque : sur-apprentissage.**

| modèle | params | mAP50 attendue | inférence CPU |
|---|---|---|---|
| yolov8n (actuel) | 3.1 M | 0.881 | ~60–120 ms |
| yolov8s | 11.2 M | 0.90–0.92 | ~200–350 ms |

Sur 864 images d'entraînement, un modèle 3,6× plus gros bénéficie surtout du transfert COCO.
À tester **sur la validation**, pas sur le test. Et attention : ton écart mAP50/mAP50-95 de
38 points suggère que le plafond est l'annotation, pas la capacité du modèle. Gain probable
plus faible que sur un dataset bruité différemment.

---

## §5 — Le seul vrai levier : la qualité des annotations

**Coût : plusieurs semaines. Gain : +5 à 10 points de mAP50-95, peu de mAP50.**

C'est la conclusion que tes chiffres imposent. L'écart mAP50 → mAP50-95 est de **38 points,
uniforme sur les 6 classes**, y compris `person` (0.896 → 0.521). Si c'était un problème de
modèle, l'écart varierait selon la classe. Sa constance prouve un **plafond d'annotation** :
les boîtes dessinées ne collent pas exactement aux objets.

Deux cibles, par ordre d'impact sur ton usage :

1. **Boîtes `vest` resserrées sur le torse** (au lieu d'aller aux genoux) → améliore le rappel
   `vest` (0.794, le 2ᵉ plus mauvais) et la précision de localisation.
2. **Casques bleus** : vérifier que les 84 instances du test ne sont pas ambiguës (casque de
   dos ? casquette bleue ?) → c'est ta pire classe en rappel.

Ne t'engage pas là-dedans sans raison : **c'est 1330 images à revoir.** Fais-le seulement si
ton application exige de recadrer les gilets (contrôle de conformité par culture d'image).

---

## §6 — Ce qui NE marchera pas (ne perds pas ton temps)

| Idée | Pourquoi c'est inutile ici |
|---|---|
| « plus d'epochs » | l'early stopping a arrêté à 63/100 : le modèle a convergé |
| « class weights » pour le déséquilibre | non supporté nativement par Ultralytics, et tes classes rares atteignent déjà 0.81–0.86 |
| « changer l'optimizer / le LR » | la loss est descendue proprement, aucun signe de instabilité |
| « plus de batch » | 16 sur T4 est déjà confortable, ça ne change pas la qualité finale |
| « plus d'augmentation » | tes courbes ne montrent aucun sur-apprentissage (val ≥ train) |

---

## Plan d'action recommandé

| Priorité | Action | Coût | Gain |
|---|---|---|---|
| **1** | `calibrate.py` + ajuster `conf` dans `predict.py` | 5 min | **+3 à 6 pts de rappel** |
| **2** | `predict.py --rule` sur des images réelles | 10 min | transforme le modèle en outil |
| **3** | réentraîner en `imgsz=960` (dataset HR) | 1 h 30 | +2 à 4 mAP50 |
| 4 | `yolov8s` en comparaison | 2 h | +2 à 4 mAP50 (à vérifier) |
| 5 | ré-annoter les gilets | semaines | +5 à 10 mAP50-95 |

Tu as déjà un modèle publiable. Les étapes 1 et 2 te donnent un **outil utilisable** en moins
d'une heure ; les étapes 3–5 sont des itérations d'ingénieur, pas des corrections de défauts.
