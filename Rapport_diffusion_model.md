# Rapport d'activité - Diffusion-model

Date: 31 mars 2026

## 1. Contexte

Ce dépôt Python implémente un modèle de diffusion DDPM avec une architecture UNet modulaire, un plan de bruit, une boucle d'entraînement et des utilitaires de génération.

L'objectif principal a été d'améliorer la robustesse du projet, d'ajouter des fonctionnalités de checkpointing, de reprise d'entraînement, de génération d'images et de GIF, ainsi que des outils d'évaluation et de comparaison de checkpoints.

## 2. Architecture du projet

Fichiers principaux:

- `diffusion_model/model.py`
- `diffusion_model/noise.py`
- `diffusion_model/scheduler.py`
- `diffusion_model/trainer.py`
- `diffusion_model/utils.py`
- `train.py`
- `sample.py`
- `evaluate.py`
- `compare_checkpoints.py`
- `README.md`
- `.gitignore`

## 3. Améliorations apportées

### 3.1. Modèle UNet

- Ajout de l'encodage temporel sinusoïdal (`TimeEmbedding`) dans `diffusion_model/model.py`.
- Renforcement de l'architecture UNet avec des blocs plus profonds et des couches de convolution supplémentaires.
- Ajout de compatibilité de chargement de checkpoints anciens via `LegacyUNet`.
- Amélioration de la robustesse de chargement de modèles avec `strict=False` et fallback pour les poids partiels.

### 3.2. Entraînement

- Ajout de checkpoints périodiques durant l'entraînement dans `diffusion_model/trainer.py`.
- Ajout de la reprise d'entraînement avec le paramètre `--resume` dans `train.py`.
- Ajout de la sauvegarde automatique d'échantillons pendant l'entraînement avec `--sample-interval`.
- Support de sous-ensembles de données CIFAR-10 via `--subset-size` pour des tests rapides.
- Gestion de l'entraînement sur CPU avec `--cpu`.

### 3.3. Génération et visualisation

- `sample.py` supporte désormais:
  - génération d'images sauvegardées (`--output-image`)
  - génération de GIFs d'animation (`--output-gif`)
  - contrôle du nombre d'échantillons et des timesteps de génération
- Ajout de la collecte d'images intermédiaires pour visualiser la progression de diffusion.

### 3.4. Évaluation et comparaison

- `evaluate.py`: évaluation d'un checkpoint sur CIFAR-10 avec métriques de reconstruction.
- `compare_checkpoints.py`: comparaison visuelle entre deux checkpoints en générant des images côte à côte.

### 3.5. Documentation et outils de projet

- Mise à jour de `README.md` avec des exemples d'utilisation pour l'entraînement, la génération, la reprise et l'évaluation.
- Ajout d'un fichier `.gitignore` pour exclure `__pycache__`, les fichiers de sortie, et les données temporaires.

## 4. Exemples de commandes utilisées

- Entraînement court:
  ```bash
  python train.py --epochs 1 --batch-size 32 --lr 2e-4 --timesteps 50 --image-size 32 --output outputs/unet_trained.pth --cpu
  ```

- Entraînement avec checkpoints et échantillons:
  ```bash
  python train.py --epochs 3 --batch-size 32 --lr 2e-4 --timesteps 100 --image-size 32 --subset-size 1000 --checkpoint-interval 1 --checkpoint-dir outputs/checkpoints --sample-interval 1 --sample-num 4 --sample-timesteps 100 --sample-output-dir outputs/train_samples --output outputs/unet_trained_medium.pth --cpu
  ```

- Entraînement prolongé:
  ```bash
  python train.py --epochs 10 --batch-size 32 --lr 2e-4 --timesteps 100 --image-size 32 --subset-size 1000 --num-workers 0 --checkpoint-interval 1 --checkpoint-dir outputs/checkpoints --sample-interval 1 --sample-num 4 --sample-timesteps 100 --sample-output-dir outputs/train_samples --output outputs/unet_trained_final.pth --cpu
  ```

## 5. Résultats générés

- Modèle sauvegardé: `outputs/unet_trained_final.pth`
- Checkpoints: `outputs/checkpoints/checkpoint_epoch_*.pth`
- Échantillons générés pendant l'entraînement: `outputs/train_samples/`
- Fichiers de sortie potentiels: `outputs/samples.png`, `outputs/samples.gif`, `outputs/checkpoint_comparison.png`

## 6. État Git et branche

- Branche active: `Abdou`
- Dernière action: commit et push des modifications vers `origin Abdou`
- Modifications principales: ajout de l'évaluation CIFAR-10, comparaison de checkpoints, UNet plus profond, reprise d'entraînement, `.gitignore`, et documentation mise à jour.

## 7. Prochaines étapes suggérées

- Entraîner plus longtemps sur le dataset complet CIFAR-10 avec `timesteps=1000`.
- Générer des GIFs pour visualiser la diffusion de plusieurs checkpoints.
- Comparer la qualité des images produites par plusieurs checkpoints avec `compare_checkpoints.py`.
- Ajouter un script d'évaluation automatique de FID/IS si nécessaire.
