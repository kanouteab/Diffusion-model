import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

from diffusion_model import load_model_from_checkpoint


def tensor_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0


def load_image_as_tensor(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(img).unsqueeze(0)  # [1, C, H, W]


def tensor_chw_to_bgr_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    img = tensor_to_01(img_tensor).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_uint8_to_tensor_chw(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    return img * 2.0 - 1.0


def bilateral_postprocess(img_tensor: torch.Tensor, d=5, sigma_color=40, sigma_space=40) -> torch.Tensor:
    img_bgr = tensor_chw_to_bgr_uint8(img_tensor)
    filtered = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return bgr_uint8_to_tensor_chw(filtered)


def denoise_iterative(model, x_t: torch.Tensor, start_t: int, device: torch.device) -> torch.Tensor:
    img = x_t.clone()

    for i in reversed(range(start_t + 1)):
        t = torch.full((img.size(0),), i, device=device, dtype=torch.long)
        predicted_noise = model(img, t)

        beta = model.betas[i]
        sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod[i]
        sqrt_recip_alpha = 1.0 / model.sqrt_alphas[i]

        if i > 0:
            posterior_variance = beta * (1.0 - model.alphas_cumprod[i - 1]) / (1.0 - model.alphas_cumprod[i])
            noise = torch.randn_like(img)
        else:
            posterior_variance = torch.tensor(0.0, device=device, dtype=img.dtype)
            noise = torch.zeros_like(img)

        mean = sqrt_recip_alpha * (
            img - beta / sqrt_one_minus_alphas_cumprod * predicted_noise
        )
        img = mean + torch.sqrt(posterior_variance) * noise

    return img.clamp(-1, 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Checkpoint introuvable: {args.model}")
    if not os.path.isfile(args.noisy_image):
        raise FileNotFoundError(f"Image bruitée introuvable: {args.noisy_image}")

    model = load_model_from_checkpoint(args.model, device, args.timesteps)
    model.eval()

    x_noisy = load_image_as_tensor(args.noisy_image, args.image_size).to(device)

    timestep = min(args.noise_timestep, model.timesteps - 1)

    with torch.no_grad():
        x_restored = denoise_iterative(model, x_noisy, timestep, device)

    if args.postprocess:
        processed = []
        for i in range(x_restored.size(0)):
            processed.append(
                bilateral_postprocess(
                    x_restored[i].detach().cpu(),
                    d=args.bilateral_d,
                    sigma_color=args.bilateral_sigma_color,
                    sigma_space=args.bilateral_sigma_space,
                )
            )
        x_restored = torch.stack(processed).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.clean_image and os.path.isfile(args.clean_image):
        x_clean = load_image_as_tensor(args.clean_image, args.image_size)
        save_image(tensor_to_01(x_clean), os.path.join(args.output_dir, "clean.png"))

    save_image(tensor_to_01(x_noisy.cpu()), os.path.join(args.output_dir, "noisy.png"))
    save_image(tensor_to_01(x_restored.cpu()), os.path.join(args.output_dir, "restored.png"))

    print(f"Résultats sauvegardés dans : {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Débruitage d'une image externe")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--noisy-image", type=str, required=True)
    parser.add_argument("--clean-image", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--noise-timestep", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="outputs/external_inference")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--bilateral-d", type=int, default=5)
    parser.add_argument("--bilateral-sigma-color", type=float, default=40.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=40.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
