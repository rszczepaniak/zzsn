from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms


class MultiClassDataset(Dataset):
    def __init__(
        self,
        image_directory,
        label_directory,
        valid_indices,
        transform=None,
        label_transform=None,
        save_data=False,
        num_classes=9,
    ):
        self.rgb_directory = os.path.join(image_directory, "rgb")
        self.nir_directory = os.path.join(image_directory, "nir")
        self.label_directory = label_directory
        self.num_classes = num_classes

        self.class_names = [
            "double_plant",
            "drydown",
            "endrow",
            "nutrient_deficiency",
            "planter_skip",
            "storm_damage",
            "water",
            "waterway",
            "weed_cluster",
        ]

        self.label_class_dirs = [
            os.path.join(label_directory, name) for name in self.class_names
        ]

        self.all_rgb_files = sorted(os.listdir(self.rgb_directory))
        self.all_nir_files = sorted(os.listdir(self.nir_directory))

        self.valid_indices = valid_indices
        self.valid_rgb_files = [self.all_rgb_files[i] for i in self.valid_indices]
        self.valid_nir_files = [self.all_nir_files[i] for i in self.valid_indices]

        self.sorted_label_files_per_class = [
            sorted(os.listdir(class_dir)) for class_dir in self.label_class_dirs
        ]
        self.valid_label_files = [
            [class_files[i] for i in self.valid_indices]
            for class_files in self.sorted_label_files_per_class
        ]

        self.transform = transform
        self.label_transform = label_transform
        self.save_data = save_data

        if self.save_data:
            os.makedirs("test_result/rgbs", exist_ok=True)
            os.makedirs("test_result/nirs", exist_ok=True)
            os.makedirs("test_result/labels", exist_ok=True)

    def __len__(self):
        return len(self.valid_rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_directory, self.valid_rgb_files[idx])
        nir_path = os.path.join(self.nir_directory, self.valid_nir_files[idx])

        rgb = Image.open(rgb_path).convert("RGB")
        nir = Image.open(nir_path)

        if self.save_data and (idx % 3 == 0):
            rgb.save(f"test_result/rgbs/{idx}.png")
            nir.save(f"test_result/nirs/{idx}.png")

        rgb_tensor = transforms.ToTensor()(rgb)
        nir_tensor = transforms.ToTensor()(nir)

        assert rgb_tensor.shape[1:] == nir_tensor.shape[1:], (
            f"RGB/NIR shape mismatch at idx {idx}"
        )
        image = torch.cat((rgb_tensor, nir_tensor), dim=0)  # [4, H, W]

        # Now apply 4-channel normalization if provided
        if self.transform:
            image = self.transform(image)

        label_tensor = []

        for class_idx in range(self.num_classes):
            label_file = self.valid_label_files[class_idx][idx]
            label_path = os.path.join(self.label_class_dirs[class_idx], label_file)
            label_img = Image.open(label_path).convert("L")

            label_array = np.array(label_img)
            binary_mask = (label_array > 0).astype(np.float32)
            label_tensor.append(torch.from_numpy(binary_mask))

        label_tensor = torch.stack(label_tensor, dim=0)  # Shape: [num_classes, H, W]
        assert label_tensor.shape[1:] == image.shape[1:], (
            f"Label/image mismatch at idx {idx}"
        )

        if self.label_transform:
            label_tensor = self.label_transform(label_tensor)

        return image, label_tensor
