import os
import torch
from torchvision.utils import save_image

from diffusion_model import load_model_from_checkpoint
from diffusion_model.noise import q_sample
from diffusion_model.utils import get_dataloader


def tensor_to_01(x):
    return (x.clamp(-1, 1) + 1) / 2


def denoise_one_step(model, x_t, t):
    predicted_noise = model(x_t, t)

    sqrt_alpha_bar = model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    x0_pred = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
    return x0_pred.clamp(-1, 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(args.model, device, args.timesteps)
    model.eval()

    dataloader = get_dataloader(
        batch_size=args.num_samples,
        image_size=args.image_size,
        train=False,
    )

    batch = next(iter(dataloader))
    x, _ = batch
    x = x.to(device)

    t = torch.full((x.size(0),), args.noise_timestep, device=device, dtype=torch.long)

    with torch.no_grad():
        x_t, _ = q_sample(
            x,
            t,
            model.sqrt_alphas_cumprod,
            model.sqrt_one_minus_alphas_cumprod,
        )
        x_restored = denoise_one_step(model, x_t, t)

    os.makedirs(args.output_dir, exist_ok=True)

    save_image(tensor_to_01(x), f"{args.output_dir}/clean.png")
    save_image(tensor_to_01(x_t), f"{args.output_dir}/noisy.png")
    save_image(tensor_to_01(x_restored), f"{args.output_dir}/restored.png")

    print("Denoising terminé !")
