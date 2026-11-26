# Note sur la licence des données

Ton dataset semble provenir de banques d'images (filigranes « alamy », « dreamstime »
visibles sur les images). Avant de publier le projet :

- **Ne pousse jamais les images sur GitHub** (le `.gitignore` les exclut déjà : `data/`,
  `datasets/`, `*.zip`).
- Si tu publies le **modèle** (`best.pt`), vérifie les conditions d'utilisation de la source.
  Un modèle entraîné sur des images sous licence est en zone grise : la plupart des
  fournisseurs interdisent l'usage commercial des images, pas l'entraînement à des fins de
  recherche. Pour un projet de portfolio, cite la source.
- Le code, lui, t'appartient : publie-le sans restriction.

Formulation prudente à mettre dans le README :

> Dataset d'entraînement non distribué (images issues de banques d'images publiques,
> utilisées à des fins de recherche). Le code et les scripts sont fournis librement.
