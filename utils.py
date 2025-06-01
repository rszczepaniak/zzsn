from torchvision import transforms
from PIL import Image
import os
import torch
import pickle
import matplotlib.pyplot as plt


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


def create_all_indices():
    for dir in os.listdir("data/supervised/Agriculture-Vision-2021/val/labels/"):
        indices = get_good_single_class_indices(
            f"data/supervised/Agriculture-Vision-2021/val/labels/{dir}"
        )
        os.makedirs("indices", exist_ok=True)
        with open(f"indices/{dir}.pkl", "wb") as f:
            print(f"Dumping indices to: {dir}.pkl")
            pickle.dump(indices, f)
