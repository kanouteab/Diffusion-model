"""Utilitaires dataset, dataloader et checkpoints."""
import os
import random
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CelebA

from .model import LegacyUNet, UNet


class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = []

        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.files.append(os.path.join(dirpath, name))

        if not self.files:
            raise FileNotFoundError(f"Aucune image trouvée dans {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def _infer_checkpoint_timesteps(state, default_timesteps):
    if isinstance(state, dict):
        betas = state.get("betas")
        if isinstance(betas, torch.Tensor) and betas.ndim == 1 and betas.numel() > 0:
            return int(betas.numel())
    return int(default_timesteps)


def load_model_from_checkpoint(checkpoint, device, timesteps):
    state = torch.load(checkpoint, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    checkpoint_timesteps = _infer_checkpoint_timesteps(state, timesteps)

    model = UNet(img_channels=3, base_channel=64, timesteps=checkpoint_timesteps).to(device)

    try:
        model.load_state_dict(state)
    except RuntimeError:
        print(f"Checkpoint incompatible avec UNet. Chargement du LegacyUNet pour {checkpoint}.")
        model = LegacyUNet(img_channels=3, base_channel=64, timesteps=checkpoint_timesteps).to(device)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


def find_kaggle_celeba_root():
    possible_paths = [
        "/kaggle/input/celeba-dataset",
        "/kaggle/input/celeba-dataset/img_align_celeba",
        "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
        "/kaggle/input/img-align-celeba",
        "/kaggle/input/celeba",
    ]

    for path in possible_paths:
        if os.path.isdir(path):
            return path

    if os.path.isdir("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return root

    return None


def get_transform(image_size=64, train=True, augment=True):
    transforms = [
        T.CenterCrop(178),
        T.Resize((image_size, image_size)),
    ]

    if train and augment:
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    transforms.extend([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return T.Compose(transforms)


def get_dataloader(
    batch_size=32,
    image_size=64,
    train=True,
    num_workers=2,
    subset_size=None,
    augment=True,
    seed=42,
):
    transform = get_transform(image_size=image_size, train=train, augment=augment)

    kaggle_root = find_kaggle_celeba_root()

    if kaggle_root is not None:
        print(f"Chargement CelebA depuis Kaggle : {kaggle_root}")
        dataset = FlatImageDataset(root=kaggle_root, transform=transform)
    else:
        print("Chargement CelebA depuis ./data")
        split = "train" if train else "valid"
        dataset = CelebA(
            root="./data",
            split=split,
            target_type="attr",
            download=True,
            transform=transform,
        )

    if subset_size is not None and subset_size > 0:
        subset_size = min(subset_size, len(dataset))
        rng = random.Random(seed + (0 if train else 999))
        indices = rng.sample(range(len(dataset)), subset_size)
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
