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

