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
    # Create the model, loss function, and optimizer
    in_channels = 4
    out_channels = 9  # Number of classes for single-class segmentation
    model = UNet(in_channels, out_channels)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    indices_type = "val" # or train

    with open(f"indices/{indices_type}/all_classes.pkl", "rb") as file:
        indices = pickle.load(file)

    random.seed(637)
    random.shuffle(indices)
    split_index = int(0.8 * len(indices))
    train(device, model, indices, split_index, optimizer, indices_type)
    test(device, model, indices, split_index, save_plots, indices_type)

    torch.save({"state_dict": model.state_dict()}, "checkpoints/final_model.pth")


def train(device, model, indices, split_index, optimizer, indices_type):
    train_dataset = MultiClassDataset(
        image_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/images",
        label_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/labels",
        valid_indices=indices[:split_index],
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        checkpt_path = (
            f"checkpoints/nir_{indices_type}_checkpoint_" + str(epoch + 1) + "_epochs"
        )
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, checkpt_path)


def test(device, model, indices, split_index, save_plots, indices_type):
    test_dataset = MultiClassDataset(
        image_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/images",
        label_directory=f"data/supervised/Agriculture-Vision-2021/{indices_type}/labels",
        valid_indices=indices[split_index:],
        transform=transforms.ToTensor(),
        save_data=save_plots,
    )

    state_dict = torch.load(f"checkpoints/nir_{indices_type}_checkpoint_1_epochs")[
        "state_dict"
    ]  # trzeba poprawić żeby dobry checkpoint się wybierał a nie '1' na stałe
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    total_test_loss = 0.0
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

    # Calculate average test loss
    average_test_loss = total_test_loss / len(test_loader)

    print(f"Average Test Loss: {average_test_loss:.4f}")

    # Define custom colors for each class
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
