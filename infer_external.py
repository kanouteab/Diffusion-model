#%%writefile infer_external.py
import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

from diffusion_model.utils import load_model_from_checkpoint


def tensor_to_01(x):
    return (x.clamp(-1, 1) + 1.0) / 2.0


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    return transform(img).unsqueeze(0)


def denoise_iterative(model, x_t, start_t, device):
    img = x_t.clone()

    for i in reversed(range(start_t + 1)):
        t = torch.full((img.size(0),), i, device=device, dtype=torch.long)
        pred = model(img, t)

        beta = model.betas[i]
        sqrt_one_minus = model.sqrt_one_minus_alphas_cumprod[i]
        sqrt_recip = 1.0 / model.sqrt_alphas[i]

        if i > 0:
            var = beta * (1 - model.alphas_cumprod[i-1]) / (1 - model.alphas_cumprod[i])
            noise = torch.randn_like(img)
        else:
            var = 0
            noise = torch.zeros_like(img)

        mean = sqrt_recip * (img - beta / sqrt_one_minus * pred)
        img = mean + torch.sqrt(torch.tensor(var, device=device)) * noise

    return img.clamp(-1,1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(args.model, device, args.timesteps)

    x_noisy = load_image(args.noisy_image, args.image_size).to(device)

    with torch.no_grad():
        x_restored = denoise_iterative(model, x_noisy, args.noise_timestep, device)

    os.makedirs(args.output_dir, exist_ok=True)

    save_image(tensor_to_01(x_noisy.cpu()), os.path.join(args.output_dir, "noisy.png"))
    save_image(tensor_to_01(x_restored.cpu()), os.path.join(args.output_dir, "restored.png"))

    if args.clean_image:
        x_clean = load_image(args.clean_image, args.image_size)
        save_image(tensor_to_01(x_clean), os.path.join(args.output_dir, "clean.png"))

    print("Résultats sauvegardés dans :", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--noisy-image", required=True)
    parser.add_argument("--clean-image", default=None)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--noise-timestep", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs/external_inference")

    args = parser.parse_args()
    main(args)
