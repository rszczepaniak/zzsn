from torch.utils.data import Dataset
from PIL import Image
import os
import torch


class SingleClassDataset(Dataset):
    def __init__(
        self,
        image_directory,
        label_directory,
        valid_indices,
        transform=None,
        save_data=False,
    ):
        self.rgb_directory = image_directory + "/rgb"
        self.nir_directory = image_directory + "/nir"
        self.label_directory = label_directory

        self.all_rgb_files = [file for file in os.listdir(self.rgb_directory)]
        self.all_nir_files = [file for file in os.listdir(self.nir_directory)]
        self.all_label_files = [file for file in os.listdir(label_directory)]

        self.valid_indices = valid_indices

        self.valid_rgb_files = [self.all_rgb_files[i] for i in self.valid_indices]
        self.valid_nir_files = [self.all_nir_files[i] for i in self.valid_indices]
        self.valid_label_files = [self.all_label_files[i] for i in self.valid_indices]

        self.transform = transform
        self.save_data = save_data

    def __len__(self):
        return len(self.valid_label_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_directory, self.valid_rgb_files[idx])
        rgb = Image.open(rgb_path).convert("RGB")

        nir_path = os.path.join(self.nir_directory, self.valid_nir_files[idx])
        nir = Image.open(nir_path)

        label_path = os.path.join(self.label_directory, self.valid_label_files[idx])
        label = Image.open(label_path)

        if self.save_data:
            if (idx % 3) == 0:
                os.makedirs("test_result", exist_ok=True)
                rgb.save(f"test_result/rgbs/{idx}.png")
                nir.save(f"test_result/nirs/{idx}.png")
                label.save(f"test_result/labels/{idx}.png")
            else:
                pass

        if self.transform:
            rgb = self.transform(rgb)
            nir = self.transform(nir)
            label = self.transform(label)

        image = torch.cat((rgb, nir), dim=0)

        return image, label
