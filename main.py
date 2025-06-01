import torch.nn as nn
from datasets.dataset import SingleClassDataset
from models.unet import UNet
from utils import config_plot
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the model, loss function, and optimizer
    in_channels = 4
    out_channels = 2  # Number of classes for single-class segmentation
    model = UNet(in_channels, out_channels)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for indices in os.listdir("indices"):
        indices_name = indices.split(".")[0]
        with open(f"indices/{indices_name}.pkl", "rb") as file:
            indices = pickle.load(file)

        random.seed(637)
        random.shuffle(indices)
        split_index = int(0.8 * len(indices))
        train(device, model, indices_name, indices, split_index, optimizer)
        test(device, model, indices_name, indices, split_index)
        break


def train(device, model, indices_name, indices, split_index, optimizer):
    train_dataset = SingleClassDataset(
        image_directory="data/supervised/Agriculture-Vision-2021/val/images",
        label_directory=f"data/supervised/Agriculture-Vision-2021/val/labels/{indices_name}",
        valid_indices=indices[:split_index],
        transform=transforms.ToTensor(),
    )

    # Add data to loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.to(device)
    num_epochs = 1
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(
                outputs, labels.squeeze(1)
            )  # Remove the channel dimension from labels
            print(loss)
            loss.backward()
            optimizer.step()
            # Move data back to CPU to save GPU memory
            # inputs, labels = inputs.cpu(), labels.cpu()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        checkpt_path = (
            f"checkpoints/nir_{indices_name}_checkpoint_" + str(epoch + 1) + "_epochs"
        )
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, checkpt_path)


def test(device, model, indices_name, indices, split_index):
    test_dataset = SingleClassDataset(
        image_directory="data/supervised/Agriculture-Vision-2021/val/images",
        label_directory=f"data/supervised/Agriculture-Vision-2021/val/labels/{indices_name}",
        valid_indices=indices[split_index:],
        transform=transforms.ToTensor(),
        save_data=True,
    )

    state_dict = torch.load(f"checkpoints/nir_{indices_name}_checkpoint_1_epochs")[
        "state_dict"
    ]  # 1 trzeba ustawić
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    model.eval()  # Set the model to evaluation mode
    model.to(device)
    total_test_loss = 0.0
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            outputs = model(inputs)

            test_loss = criterion(outputs, labels.squeeze(1))
            total_test_loss += test_loss.item()

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
    preds = []
    gts = []
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model.cpu()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            gts.append(labels)
            outputs = model(inputs)
            labels = labels.long()
            # Convert output to probabilities and get the predicted class
            _, predicted = torch.max(outputs, 1)

            # Display the original image, ground truth, and predicted segmentation mask
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Original image
            axes[0].imshow(F.to_pil_image(inputs[0]))  # Assuming batch_size=1
            axes[0].set_title("Original Image")
            config_plot(axes[0])

            # Ground truth label
            axes[1].imshow(
                labels[0].squeeze(0), cmap=custom_cmap, vmin=0, vmax=9
            )  # Assuming single-channel labels
            axes[1].set_title("Ground Truth")
            config_plot(axes[1])

            # Predicted segmentation mask
            axes[2].imshow(
                predicted[0].squeeze(0), cmap=custom_cmap, vmin=0, vmax=9
            )  # Adjust cmap as needed
            axes[2].set_title("Predicted Mask")

            preds.append(predicted)

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

            config_plot(axes[2])
            plt.show()
            print("Loss:", criterion(outputs, labels.squeeze(1)).item())


if __name__ == "__main__":
    main()
