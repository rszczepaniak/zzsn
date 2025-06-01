import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class AgriVisionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "images", "rgb")
        self.nir_dir = os.path.join(root_dir, "images", "nir")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.samples = sorted(os.listdir(self.rgb_dir))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file = self.samples[idx]
        name = os.path.splitext(file)[0]

        # Load 3-channel RGB image
        rgb = Image.open(os.path.join(self.rgb_dir, name + ".jpg")).convert("RGB")
        nir = Image.open(os.path.join(self.nir_dir, name + ".jpg")).convert(
            "L"
        )  # single channel

        # Combine into 4-channel image
        rgb = np.array(rgb)
        nir = np.array(nir)[..., None]
        image = np.concatenate([rgb, nir], axis=-1)  # H x W x 4

        # Load mask (H x W) with integer labels [0–7]
        mask = Image.open(os.path.join(self.mask_dir, name + ".png"))
        mask = np.array(mask).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
