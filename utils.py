import shutil

from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def read_class_labels(class_name, i, start, end):
    class_labels = []
    folder_path = "supervised/Agriculture-Vision-2021/val/labels/" + class_name
    transform = transforms.ToTensor()

    bad_prefixes = {"set!"}
    good_prefixes = {"set!"}
    num_labels = 0
    for file in os.listdir(folder_path)[start:end]:
        prefix = file.split("_")[0]

        if prefix in bad_prefixes:  # if we know it'll be all zeroes
            class_labels.append(torch.zeros(512, 512))

        else:  # we open it because it's either an actual label or we don't know yet
            try:
                label = Image.open(os.path.join(folder_path, file))
                label = transform(label) * i

            except (Image.UnidentifiedImageError, OSError) as e:
                print(f"Error opening image {file}: {e}")
            if prefix in good_prefixes:  # if we know it's an actual label
                class_labels.append(label)
                num_labels += 1

            else:  # if it's a new prefix we haven't seen:
                max_val = torch.max(label)
                if max_val == 0:
                    bad_prefixes.add(prefix)
                else:
                    good_prefixes.add(prefix)
                    num_labels += 1

                class_labels.append(label)

    return class_labels, num_labels


def get_good_single_class_indices(folder_path):
    good_label_indices = []
    bad_prefixes = {"set!"}
    good_prefixes = {"set!"}

    transform = transforms.ToTensor()

    for i in range(len(os.listdir(folder_path))):
        file = os.listdir(folder_path)[i]
        prefix = file.split("_")[0]
        if (i % 100) == 0:
            print(i)
        if prefix in bad_prefixes:  # if we know it'll be all zeroes
            continue

        elif prefix in good_prefixes:  # if we know it's an actual label
            good_label_indices.append(i)
            continue

        else:  # we open it because we don't know yet
            try:
                label = Image.open(os.path.join(folder_path, file))
                label = transform(label)

            except (Image.UnidentifiedImageError, OSError) as e:
                print(f"Error opening image {file}: {e}")

            # it's a new prefix we haven't seen, so check if it's good or bad
            max_val = torch.max(label)
            if max_val == 0:
                bad_prefixes.add(prefix)
            else:
                good_prefixes.add(prefix)
                good_label_indices.append(i)

    return good_label_indices


def get_valid_multiclass_indices(label_root):
    transform = transforms.ToTensor()
    all_class_dirs = sorted(os.listdir(label_root))
    num_files = len(os.listdir(os.path.join(label_root, all_class_dirs[0])))

    good_indices = []

    for i in range(num_files):
        found_positive = False

        for class_dir in all_class_dirs:
            class_path = os.path.join(label_root, class_dir)
            file = sorted(os.listdir(class_path))[i]

            try:
                label = Image.open(os.path.join(class_path, file))
                label_tensor = transform(label)
                if label_tensor.max() > 0:
                    found_positive = True
                    break  # no need to check other classes

            except (Image.UnidentifiedImageError, OSError) as e:
                print(f"Error reading file {file}: {e}")
                continue

        if found_positive:
            good_indices.append(i)

        if i % 100 == 0:
            print(f"Checked {i}/{num_files}")

    return good_indices


def plot(tensor):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(tensor)
    plt.show()


def config_plot(ax):
    """
    Function to remove axis tickers and box around figure
    """

    ax.axis("off")

    # Remove axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)


def create_all_indices(indices_type="val"):
    for dir in os.listdir(
        f"data/supervised/Agriculture-Vision-2021/{indices_type}/labels/"
    ):
        indices = get_good_single_class_indices(
            f"data/supervised/Agriculture-Vision-2021/{indices_type}/labels/{dir}"
        )
        os.makedirs("indices", exist_ok=True)
        with open(f"indices/{indices_type}/{dir}.pkl", "wb") as f:
            print(f"Dumping indices to: {dir}.pkl")
            pickle.dump(indices, f)


def create_multiclass_indices(indices_type="val"):
    label_root = f"data/supervised/Agriculture-Vision-2021/{indices_type}/labels"
    indices = get_valid_multiclass_indices(label_root)
    os.makedirs("indices", exist_ok=True)
    with open(f"indices/{indices_type}/all_classes.pkl", "wb") as f:
        pickle.dump(indices, f)
    print(
        f"Saved {len(indices)} valid multi-class indices to indices/{indices_type}/all_classes.pkl"
    )


def compute_class_pos_weights(dataset, num_classes=9):
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )

    positive_pixel_counts = torch.zeros(num_classes)
    total_pixel_count = 0

    for _, labels in loader:
        B, C, H, W = labels.shape

        positive_pixel_counts += labels.sum(
            dim=(0, 2, 3)
        )  # Sum over batch + spatial dims
        total_pixel_count += B * H * W

    neg_pixel_counts = total_pixel_count - positive_pixel_counts
    pos_weights = neg_pixel_counts / (positive_pixel_counts + 1e-6)

    return pos_weights


class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input, target):
        # input, target: [B, C, H, W]
        B, C, H, W = input.shape
        pw = self.pos_weight.view(1, C, 1, 1)
        return nn.functional.binary_cross_entropy_with_logits(
            input, target, pos_weight=pw
        )


def custom_collate_fn(batch):
    all_tiles = []
    all_positions = []
    all_shapes = []

    for sample in batch:
        all_tiles.extend(sample["tiles"])
        all_positions.extend(sample["positions"])
        all_shapes.append(sample["shape"])

    if all_tiles:
        tiles_tensor = torch.stack(all_tiles)
    else:
        tiles_tensor = torch.empty((0, 4, 512, 512))  # Empty placeholder

    return tiles_tensor, all_positions, all_shapes


def generate_pseudo_labels(model, unlabeled_dataset, device, threshold=0.9, output_dir="pseudo_data"):
    model.eval()
    model.to(device)

    loader = DataLoader(
        unlabeled_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=False,
    )
    print("Loader created")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (tiles, tile_positions, sizes) in enumerate(loader):
            if tiles.shape[0] == 0:
                continue

            height, width = sizes[0]  # Not used, but kept for future spatial aggregation if needed
            sub_batch_size = 8

            for i in range(0, len(tiles), sub_batch_size):
                sub_tiles = tiles[i:i + sub_batch_size].to(device)  # [B, 4, 512, 512]
                sub_positions = tile_positions[i:i + sub_batch_size]

                outputs = model(sub_tiles)  # [B, 9, 512, 512]
                probs = torch.sigmoid(outputs)
                mask = probs > threshold  # [B, 9, 512, 512]

                for j, (y, x, y_end, x_end) in enumerate(sub_positions):
                    tile_mask = mask[j].float()  # [9, 512, 512]

                    # Ensure there's at least some confident class
                    if tile_mask.sum(dim=(1, 2)).max() == 0:
                        continue

                    output_path = os.path.join(output_dir, f"sample_{idx}_tile_{i + j}.pt")
                    torch.save({"image": sub_tiles[j], "mask": tile_mask}, output_path)
