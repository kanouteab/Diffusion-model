"""Script de génération d'images à partir d'un modèle de diffusion entraîné."""
import argparse
import os

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from diffusion_model import load_model_from_checkpoint
from diffusion_model.noise import p_sample_loop


def main(args):
    # Choix du device: GPU si disponible, sinon CPU.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Vérification du checkpoint avant tentative de chargement.
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Checkpoint introuvable: {args.model}")

    # Affichage du device pour tracer l'environnement d'exécution.
    print(f"Device utilisé: {device}")
    # Chargement robuste (UNet / LegacyUNet) avec adaptation possible des timesteps.
    model = load_model_from_checkpoint(args.model, device, args.timesteps)
    # Seed fixe pour reproductibilité de l'échantillonnage.
    torch.manual_seed(args.seed)
    # On ne dépasse jamais le nombre de timesteps supporté par le checkpoint.
    effective_timesteps = min(args.timesteps, model.timesteps)
    if effective_timesteps != args.timesteps:
        print(
            f"Timesteps ajustés de {args.timesteps} à {effective_timesteps} "
            "pour correspondre au checkpoint."
        )

    # Génération sans gradients pour réduire la mémoire et accélérer.
    with torch.no_grad():
        if args.output_gif:
            # Mode GIF: on récupère aussi des frames intermédiaires.
            out, frames = p_sample_loop(
                model,
                (args.num_samples, 3, args.image_size, args.image_size),
                effective_timesteps,
                device,
                return_intermediates=True,
                frame_interval=args.gif_frame_interval,
            )
        else:
            # Mode standard: on retourne uniquement les échantillons finaux.
            out = p_sample_loop(
                model,
                (args.num_samples, 3, args.image_size, args.image_size),
                effective_timesteps,
                device,
            )
    # Normalisation vers [0, 1] pour sauvegarde/visualisation.
    out = (out.clamp(-1, 1) + 1) / 2.0

    # Création du dossier de sortie des tenseurs si besoin.
    # Si aucun dossier n'est fourni, on sauvegarde dans le dossier courant.
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    # Sauvegarde du lot d'images sous forme tensorielle PyTorch.
    torch.save(out, args.output)

    # Chemin image par défaut: même nom que .pt mais extension .png.
    image_path = args.output_image or os.path.splitext(args.output)[0] + ".png"
    # Sauvegarde d'une grille PNG des échantillons.
    save_image(out, image_path, nrow=min(8, args.num_samples))

    print(f"Échantillons générés: {args.output}")
    print(f"Image sauvegardée: {image_path}")

    if args.output_gif:
        # Chemin de destination du GIF fourni via CLI.
        gif_path = args.output_gif
        # Création du dossier GIF si nécessaire.
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        grid_frames = []
        for frame in frames:
            # Conversion de chaque frame intermédiaire en grille RGB PIL.
            grid = make_grid((frame.clamp(-1, 1) + 1) / 2.0, nrow=min(8, args.num_samples))
            ndarr = (grid.mul(255).permute(1, 2, 0).byte().cpu().numpy())
            grid_frames.append(Image.fromarray(ndarr))
        # Écriture du GIF animé avec fps configurable.
        grid_frames[0].save(
            gif_path,
            save_all=True,
            append_images=grid_frames[1:],
            duration=int(1000 / max(1, args.gif_fps)),
            loop=0,
        )
        print(f"GIF sauvegardé: {gif_path}")


if __name__ == "__main__":
    # Définition des arguments CLI du script de sampling.
    parser = argparse.ArgumentParser(description="Génère des images par diffusion")
    # Checkpoint (obligatoire) des poids à charger.
    parser.add_argument("--model", type=str, required=True)
    # Fichier .pt qui contiendra le tenseur des échantillons.
    parser.add_argument("--output", type=str, default="samples.pt")
    # Image PNG de sortie (optionnelle, auto-calculée si absente).
    parser.add_argument("--output-image", type=str, default=None)
    # GIF de diffusion intermédiaire (optionnel).
    parser.add_argument("--output-gif", type=str, default=None)
    # Fréquence d'échantillonnage des frames intermédiaires.
    parser.add_argument("--gif-frame-interval", type=int, default=10)
    # Vitesse du GIF en images par seconde.
    parser.add_argument("--gif-fps", type=int, default=5)
    # Nombre d'étapes de débruitage demandées.
    parser.add_argument("--timesteps", type=int, default=1000)
    # Taille spatiale des images générées.
    parser.add_argument("--image-size", type=int, default=32)
    # Nombre d'images à générer dans le lot.
    parser.add_argument("--num-samples", type=int, default=8)
    # Seed globale pour reproductibilité.
    parser.add_argument("--seed", type=int, default=42)
    # Forcer l'exécution CPU.
    parser.add_argument("--cpu", action="store_true")
    # Parsing effectif des arguments CLI.
    args = parser.parse_args()
    # Point d'entrée principal du script.
    main(args)
