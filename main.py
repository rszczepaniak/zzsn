import torch.nn as nn
from datasets.dataset import MultiClassDataset
from models.unet import UNet
from utils import config_plot, create_all_indices, create_multiclass_indices
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch
import random
import pickle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def main(save_plots=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 4
    out_channels = 9
    model = UNet(in_channels, out_channels)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    indices_type = "val"

    with open(f"indices/{indices_type}/all_classes.pkl", "rb") as file:
        indices = pickle.load(file)

    random.seed(637)
    random.shuffle(indices)
    total = len(indices)
    train_split = int(0.7 * total)
    val_split = int(0.85 * total)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    input_transform = transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)

    train_dataset = MultiClassDataset(
        image_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/images",
        label_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/labels",
        valid_indices=train_indices,
        transform=input_transform,
    )
    val_dataset = MultiClassDataset(
        image_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/images",
        label_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/labels",
        valid_indices=val_indices,
        transform=input_transform,
    )
    test_dataset = MultiClassDataset(
        image_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/images",
        label_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/labels",
        valid_indices=test_indices,
        transform=input_transform,
        save_data=save_plots,
    )

    best_model_path = train(device, model, optimizer, train_dataset, val_dataset)
    test(device, model, test_dataset, best_model_path, save_plots, indices_type)


def train(device, model, optimizer, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    best_val_acc = 0.0
    best_model_path = "checkpoints/best_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(10):
        model.train()
        correct_pixels = 0
        total_pixels = 0
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                true_labels = torch.argmax(labels, dim=1)
                correct_pixels += (predicted == true_labels).sum().item()
                total_pixels += true_labels.numel()

        train_acc = correct_pixels / total_pixels
        avg_train_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/10]\nTrain Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}"
        )

        # Validate after each epoch
        val_acc, val_loss = validate(device, model, val_dataset)

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict()}, best_model_path)

        # Optionally save per-epoch model
        torch.save(
            {"state_dict": model.state_dict()},
            f"checkpoints/model_epoch_{epoch + 1}.pth",
        )

    return best_model_path


def validate(
    device, model, val_dataset, model_path=None, save_plots=False, indices_type=None
):
    if model_path:
        model.load_state_dict(torch.load(model_path)["state_dict"])
    model.to(device)
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(labels, dim=1)
            correct_pixels += (predicted == true_labels).sum().item()
            total_pixels += true_labels.numel()

    acc = correct_pixels / total_pixels
    avg_loss = total_loss / len(val_loader)

    if save_plots and indices_type:
        save_visualizations(model.cpu(), indices_type)

    return acc, avg_loss


def test(device, model, test_dataset, model_path, save_plots=False, indices_type=None):
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(labels, dim=1)
            correct_pixels += (predicted == true_labels).sum().item()
            total_pixels += true_labels.numel()

    acc = correct_pixels / total_pixels
    avg_loss = total_loss / len(test_loader)

    print(f"\n=== Final Test Results ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")

    if save_plots and indices_type:
        save_visualizations(model.cpu(), test_dataset, indices_type)


def save_visualizations(model, indices_type):
    custom_colors = [
        "#1f77b4",
        "#bcbd22",
        "#2ca02c",
        "#ff7f0e",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#17becf",
    ]

    # Create a new colormap with custom colors
    custom_cmap = ListedColormap(custom_colors, name="custom_colormap")

    model.eval()
    gts = []
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model.cpu()

    os.makedirs("results", exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            gts.append(labels)
            outputs = model(inputs)
            labels = labels.long()
            _, predicted = torch.max(outputs, 1)

            if save_plots:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(F.to_pil_image(inputs[0]))
                axes[0].set_title("Original Image")
                config_plot(axes[0])

                axes[1].imshow(labels[0].squeeze(0), cmap=custom_cmap, vmin=0, vmax=9)
                axes[1].set_title("Ground Truth")
                config_plot(axes[1])

                axes[2].imshow(
                    predicted[0].squeeze(0), cmap=custom_cmap, vmin=0, vmax=9
                )
                axes[2].set_title("Predicted Mask")
                config_plot(axes[2])

                legend_labels = ["Background", "Water Area"]
                plt.legend(
                    handles=[
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=10,
                            label=label,
                        )
                        for color, label in zip(custom_colors, legend_labels)
                    ],
                    title="Legend",
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                )

                # Save figure to results/
                filename = f"results/{indices_type}_{i}.png"
                plt.savefig(filename, bbox_inches="tight")
                plt.close(fig)

                print("Saved figure to:", filename)
            print("Loss:", criterion(outputs, labels.squeeze(1)).item())


if __name__ == "__main__":
    # create_all_indices()
    # create_multiclass_indices("train")
    main(save_plots=True)
