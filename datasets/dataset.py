from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms
import math
import rasterio  # Zastąp imageio na rasterio dla dużych TIFF-ów


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


class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None, tile_size=512, stride=512):
        self.root_dir = root_dir
        self.transform = transform
        self.tile_size = tile_size
        self.stride = stride
        self.sample_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]
        imagery_dir = os.path.join(sample_path, "imagery")

        # Ensure all files exist
        required_files = ["red.tif", "green.tif", "blue.tif", "nir.tif"]
        for fname in required_files:
            full_path = os.path.join(imagery_dir, fname)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"Missing {fname} in {imagery_dir}")

        try:
            with (
                rasterio.open(os.path.join(imagery_dir, "red.tif")) as red_ds,
                rasterio.open(os.path.join(imagery_dir, "green.tif")) as green_ds,
                rasterio.open(os.path.join(imagery_dir, "blue.tif")) as blue_ds,
                rasterio.open(os.path.join(imagery_dir, "nir.tif")) as nir_ds,
            ):
                height, width = red_ds.height, red_ds.width

                tiles = []
                tile_positions = []

                for y in range(0, height, self.stride):
                    for x in range(0, width, self.stride):
                        y_end = min(y + self.tile_size, height)
                        x_end = min(x + self.tile_size, width)
                        if y_end - y < self.tile_size or x_end - x < self.tile_size:
                            continue

                        window = rasterio.windows.Window(x, y, x_end - x, y_end - y)

                        def read_and_norm(ds):
                            band = ds.read(1, window=window).astype(np.float32)
                            max_val = band.max()
                            if max_val == 0:
                                return band  # or raise Warning?
                            return band / max_val if max_val > 1 else band

                        tile = torch.stack(
                            [
                                torch.from_numpy(read_and_norm(red_ds)),
                                torch.from_numpy(read_and_norm(green_ds)),
                                torch.from_numpy(read_and_norm(blue_ds)),
                                torch.from_numpy(read_and_norm(nir_ds)),
                            ],
                            dim=0,
                        )

                        assert tile.shape == (4, self.tile_size, self.tile_size), (
                            f"Invalid tile shape: {tile.shape}"
                        )

                        if self.transform:
                            tile = self.transform(tile)

                        tiles.append(tile)
                        tile_positions.append((y, x, y_end, x_end))

            if not tiles:
                raise ValueError(f"No valid tiles found in {sample_path}")

            return {
                "tiles": tiles,
                "positions": tile_positions,
                "shape": (height, width),
                "sample_path": sample_path,
            }
        except rasterio.errors.RasterioIOError as e:
            print(f"⚠️ Skipping corrupted file in {sample_path}: {e}")
            return {
                "tiles": [],
                "positions": [],
                "shape": (0, 0),
                "sample_path": sample_path,
            }


class PseudoLabeledDataset(Dataset):
    def __init__(self, pseudo_data_paths, transform=None):
        self.pseudo_data_paths = pseudo_data_paths
        self.transform = transform

    def __len__(self):
        return len(self.pseudo_data_paths)

    def __getitem__(self, idx):
        data = torch.load(self.pseudo_data_paths[idx], map_location="cpu")
        image = data["image"]
        mask = data["mask"]

        assert image.shape == (4, 512, 512), (
            f"Invalid pseudo image shape: {image.shape}"
        )
        assert mask.shape == (9, 512, 512), f"Invalid pseudo mask shape: {mask.shape}"

        if self.transform:
            image = self.transform(image)

        return image, mask
