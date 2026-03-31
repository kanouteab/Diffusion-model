# Diffusion-model
<p align="center">
  <img src="assets/logo_m.png" width="300"/>
</p>

## État actuel
- Pas de fichier PDF dans `assets/` (seulement `logo.png` et `logo_m.png`).
- Le contenu attendu du PDF n'est pas disponible.

## Architecture modulaire ajoutée
- `diffusion_model/scheduler.py`: plan de bruit linéaire
- `diffusion_model/noise.py`: logique d'échantillonnage DDPM
- `diffusion_model/model.py`: UNet minimal
- `diffusion_model/trainer.py`: boucle d'entraînement
- `diffusion_model/utils.py`: accès aux datasets (CIFAR10)
- `train.py`: script d'entraînement CLI
- `sample.py`: script d’échantillonnage CLI

## Instructions rapides
1. Installer les dépendances:
```bash
pip install -r requirements.txt
```
2. Lancer l'entraînement:
```bash
python train.py --epochs 20 --batch-size 128
```
3. Générer des échantillons:
```bash
python sample.py --model diffusion_model.pth --output samples.pt
```

## Nouvelles fonctionnalités ajoutées
- Checkpoints périodiques pendant l'entraînement avec `--checkpoint-interval` et `--checkpoint-dir`
- Sauvegarde automatique d'échantillons pendant l'entraînement via `--sample-interval`
- Génération d'images en grille avec `--output-image`
- Génération de GIFs de diffusion avec `--output-gif` et `--gif-frame-interval`

### Exemples
Entraîner sur un petit sous-ensemble et enregistrer des checkpoints et des images:
```bash
python train.py --epochs 5 --batch-size 32 --subset-size 5000 --checkpoint-interval 1 --checkpoint-dir outputs/checkpoints --sample-interval 1 --sample-num 4 --sample-output-dir outputs/train_samples --cpu
```
Reprendre l'entraînement depuis un checkpoint existant:
```bash
python train.py --resume outputs/checkpoints/checkpoint_epoch_1.pth --epochs 5 --batch-size 32 --subset-size 5000 --checkpoint-interval 1 --checkpoint-dir outputs/checkpoints --sample-interval 1 --sample-num 4 --sample-output-dir outputs/train_samples --cpu
```
Évaluer un modèle sur un ensemble de validation CIFAR-10:
```bash
python evaluate.py --model outputs/unet_trained_final.pth --timesteps 100 --batch-size 64 --subset-size 1000 --num-batches 10 --cpu
```
Comparer deux checkpoints par génération d'images:
```bash
python compare_checkpoints.py --checkpoints outputs/checkpoints/checkpoint_epoch_5.pth outputs/checkpoints/checkpoint_epoch_10.pth --output-image outputs/checkpoint_comparison.png --timesteps 100 --num-samples 4 --cpu
```
Générer des images et un GIF à partir d'un checkpoint:
```bash
python sample.py --model outputs/unet_trained.pth --output outputs/samples.pt --output-image outputs/samples.png --output-gif outputs/samples.gif --gif-frame-interval 10 --gif-fps 5 --num-samples 4 --cpu
```

