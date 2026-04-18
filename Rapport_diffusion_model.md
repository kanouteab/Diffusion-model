# Rapport final - Diffusion-model

Date : 1 avril 2026

## 1. Introduction

Ce rapport final présente les résultats et les choix techniques du projet `Diffusion-model`, une implémentation d'un modèle de diffusion (DDPM) sur le dataset CIFAR-10.

L'objectif est de construire un pipeline complet : entraînement, sauvegarde de checkpoints, reprise d'entraînement, génération d'images, évaluation et comparaison de modèles.

## 2. Objectifs du projet

- Implémenter un modèle de diffusion basé sur une architecture UNet.
- Permettre l'entraînement avec checkpoints et la reprise à partir d'un checkpoint.
- Générer des images et des GIFs pour observer l'évolution de la diffusion inversée.
- Évaluer la qualité du modèle sur un sous-ensemble CIFAR-10.
- Documenter le projet et fournir un rapport final.

## 3. Méthodologie

### 3.1. Jeu de données

Le projet utilise CIFAR-10, un ensemble standard de 60 000 images couleur 32×32 réparties en 10 classes.

Le chargement des données est géré par `diffusion_model/utils.py` avec des options :

- `--subset-size` pour accélérer les tests,
- `--image-size` pour forcer la taille des images si nécessaire,
- `--num-workers` pour le chargement multi-processus.

### 3.2. Architecture du modèle

L'architecture UNet est définie dans `diffusion_model/model.py`.

Principales fonctionnalités :

- Encodage temporel sinusoïdal pour intégrer le pas de diffusion,
- Blocs convolutifs en encodeur et décodeur,
- Skip connections pour préserver les détails spatiaux,
- Support de chargement de checkpoints plus anciens avec `LegacyUNet`.

### 3.3. Plan de diffusion

La logique de diffusion et de débruitage est répartie entre :

- `diffusion_model/noise.py` : gestion du bruit direct et de l'échantillonnage,
- `diffusion_model/scheduler.py` : définition du plan de bruit linéaire,
- `diffusion_model/trainer.py` : boucle d'entraînement et calcul des pertes.

### 3.4. Entraînement et objets sauvegardés

Le script `train.py` orchestre :

- l'initialisation du modèle,
- le chargement des données,
- la sauvegarde régulière de checkpoints,
- la génération d'échantillons intermédiaires.

Paramètres importants :

- `--epochs`,
- `--batch-size`,
- `--lr`,
- `--timesteps`,
- `--checkpoint-interval`,
- `--sample-interval`,
- `--resume`.

## 4. Résultats obtenus

### 4.1. Artefacts produits

Le projet a généré les éléments suivants :

- Modèle final : `outputs/unet_trained_final.pth`
- Checkpoints : `outputs/checkpoints/checkpoint_epoch_*.pth`
- Échantillons d'entraînement : `outputs/train_samples/`
- Rapport final : `Rapport_diffusion_model.md` et `Rapport_diffusion_model.pdf`

### 4.2. Évaluation

La métrique d'évaluation implémentée est la perte MSE moyenne sur des images bruitées et reconstruites, calculée par `evaluate.py`.

- Résultat clé à insérer : MSE moyen final sur le sous-ensemble CIFAR-10.

### 4.3. Analyse qualitative

La génération d'images est testée avec `sample.py` et `compare_checkpoints.py`, ce qui permet de visualiser :

- la qualité des images produites,
- l'effet des différents checkpoints,
- la progression de la diffusion inverse.

## 5. Figures et emplacements

### Figure 1 — Courbe d'apprentissage

Insérer ici la courbe de perte d'entraînement (loss) en fonction des epochs.

> [Figure 1 : Courbe d'apprentissage – perte d'entraînement vs époque]

### Figure 2 — Évolution de la production d'images

Insérer ici une figure montrant des images générées à différents epochs ou à différents checkpoints.

> [Figure 2 : Échantillons générés à différents checkpoints / epochs]

### Figure 3 — Évaluation CIFAR-10

Insérer ici la courbe de la métrique MSE sur le sous-ensemble de validation.

> [Figure 3 : Perte MSE moyenne sur l'évaluation CIFAR-10]

### Figure 4 — Comparaison de checkpoints

Insérer ici une grille comparant deux ou plusieurs checkpoints avec `compare_checkpoints.py`.

> [Figure 4 : Comparaison visuelle de checkpoints]

## 6. Interprétation préliminaire

- La courbe d'apprentissage devrait montrer une décroissance stable de la perte, indiquant que le modèle apprend à prédire le bruit.
- La comparaison des checkpoints permet d'identifier à quel moment la qualité d'image se stabilise.
- La métrique MSE seule ne suffit pas : il faudra compléter avec une analyse visuelle et, potentiellement, FID/IS à l'avenir.

## 7. Points forts du projet

- Pipeline complet : entraînement, sauvegarde, reprise, génération et évaluation.
- Architecture UNet modulable et compatible avec différents timesteps.
- Support de la génération d'images et de GIFs.
- Documentation claire dans `README.md`.
- Rapport final disponible pour clôturer le projet.

## 8. Limites et perspectives

- Le dataset CIFAR-10 est de petite résolution, donc les résultats ne sont pas comparables aux grands modèles de diffusion sur haute résolution.
- L'évaluation est actuellement limitée à la MSE ; il est recommandé d'ajouter un calcul de FID/IS pour une mesure de qualité plus robuste.
- La génération de GIFs et la comparaison de checkpoints peuvent être automatisées davantage.

## 9. Prochaines étapes

1. Générer les figures manquantes : courbe d'entraînement, MSE d'évaluation, grilles de génération, comparaison de checkpoints.
2. Ajouter un script automatique pour produire un PDF final à partir du Markdown.
3. Intégrer une évaluation perceptuelle (FID/IS) si les ressources GPU le permettent.
4. Documenter les commandes exactes utilisées pour chaque expérience finale.

---

*Note : ce rapport est structuré pour devenir le document final du projet. Les emplacements des figures sont indiqués ci-dessus et seront complétés après production des graphiques.*
