import random
import pathlib
import torch.nn as nn
from torchmetrics.classification import MultilabelF1Score, MultilabelJaccardIndex
from datasets.dataset import MultiClassDataset, UnlabeledDataset, PseudoLabeledDataset
from models.unet import UNet
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import os
import torch
import pickle
import json
from datetime import datetime

from utils import generate_pseudo_labels, predict_and_overlay
from plot import plot_metrics_on_same_plot


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 4
    out_channels = 9
    learning_rate = 0.0004
    batch_size = 32
    epochs = 15
    number_of_pseudo_labels_training_epochs = 2

    log = {
        "hyperparameters": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)",
            "batch_size": batch_size,
            "num_epochs": epochs,
            "num_pseudo_epochs": number_of_pseudo_labels_training_epochs,
        },
        "train_history": [],
        "final_results": {},
    }

    model = UNet(in_channels, out_channels)
    model.load_state_dict(torch.load("checkpoints/best_model_20250605_002445.pth")["state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

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
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/training_log_{timestamp}.json"

    # best_model_path = train(
    #     device,
    #     model,
    #     optimizer,
    #     scheduler,
    #     train_dataset,
    #     val_dataset,
    #     log,
    #     timestamp,
    # )

    print("\n=== Kafelkowanie ===")
    unlabeled_dataset = UnlabeledDataset(
        root_dir="data/raw",
        transform=input_transform,
        tile_size=512,
        stride=512,
    )

    for i in range(number_of_pseudo_labels_training_epochs):
        print(
            f"\n=== Generowanie pseudoetykiet z danych nieoznakowanych: [{i + 1}/{number_of_pseudo_labels_training_epochs}] ==="
        )
        generate_pseudo_labels(
            model, unlabeled_dataset, device, 0.9, "pseudo_data", timestamp
        )
        pseudo_data_paths = list(
            pathlib.Path(os.path.join("pseudo_data", timestamp)).iterdir()
        )
        print("\n=== Skończono generację pseudoetykiet z danych nieoznakowanych ===")
        if len(pseudo_data_paths) > 0:
            pseudo_dataset = PseudoLabeledDataset(
                pseudo_data_paths, transform=input_transform
            )
            combined_dataset = ConcatDataset([train_dataset, pseudo_dataset])
        else:
            combined_dataset = train_dataset
            print("Brak próbek spełniających próg ufności.\n")

        best_model_path = train(
            device,
            model,
            optimizer,
            scheduler,
            combined_dataset,
            val_dataset,
            log,
            timestamp,
        )

    test(device, model, test_dataset, best_model_path, log, False)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {log_path}")


def train(
    device,
    model,
    optimizer,
    scheduler,
    train_dataset,
    val_dataset,
    log,
    timestamp,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=log["hyperparameters"]["batch_size"],
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    best_val_acc = 0.0
    best_model_path = f"checkpoints/best_model_{timestamp}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(log["hyperparameters"]["num_epochs"]):
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
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                true_labels = labels.long()
                correct_pixels += (predicted == true_labels).sum().item()
                total_pixels += true_labels.numel()

        train_acc = correct_pixels / total_pixels
        avg_train_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{log['hyperparameters']['num_epochs']}]\nTrain Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}"
        )

        # Validate after each epoch
        val_acc, val_loss, val_f1, val_iou = validate(device, model, val_dataset, log)

        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val IoU: {val_iou:.4f}\n"
        )

        scheduler.step(val_loss)

        log["train_history"].append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1_score": val_f1,
                "val_iou": val_iou,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict()}, best_model_path)

    return best_model_path


def validate(device, model, val_dataset, log):
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=log["hyperparameters"]["batch_size"],
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Initialize GPU-based metrics
    f1_metric = MultilabelF1Score(num_labels=9, average="macro").to(device)
    iou_metric = MultilabelJaccardIndex(num_labels=9, average="macro").to(device)

    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).long()
            true_labels = labels.long()

            # Accuracy
            correct_pixels += (predicted == true_labels).sum().item()
            total_pixels += true_labels.numel()

            # Update torchmetrics
            f1_metric.update(predicted, true_labels)
            iou_metric.update(predicted, true_labels)

    acc = correct_pixels / total_pixels
    avg_loss = total_loss / len(val_loader)
    f1 = f1_metric.compute().item()
    iou = iou_metric.compute().item()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return acc, avg_loss, f1, iou


def test(device, model, test_dataset, model_path, log=None, load_model=False):
    if load_model:
        model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True
    )
    criterion = nn.BCEWithLogitsLoss()

    # GPU-based metrics
    f1_metric = MultilabelF1Score(num_labels=9, average="macro").to(device)
    iou_metric = MultilabelJaccardIndex(num_labels=9, average="macro").to(device)

    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).long()
            true_labels = labels.long()

            correct_pixels += (predicted == true_labels).sum().item()
            total_pixels += true_labels.numel()

            f1_metric.update(predicted, true_labels)
            iou_metric.update(predicted, true_labels)

    acc = correct_pixels / total_pixels
    avg_loss = total_loss / len(test_loader)
    f1 = f1_metric.compute().item()
    iou = iou_metric.compute().item()

    if log is not None:
        log["final_results"] = {
            "test_loss": avg_loss,
            "test_accuracy": acc,
            "test_f1_score": f1,
            "test_iou": iou,
        }

    print(f"\n=== Final Test Results ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test IoU: {iou:.4f}")


if __name__ == "__main__":
    # create_all_indices()
    # create_multiclass_indices("train")
    # main()
    # with open("results/training_log_20250606_080718.json") as fh:
    #     data = json.load(fh)
    # plot_metrics_on_same_plot(data, "pseudo_data")
    class_names = [
        "double_plant", "drydown", "endrow", "nutrient_deficiency",
        "planter_skip", "storm_damage", "water", "waterway", "weed_cluster"
    ]

    overlay = predict_and_overlay(
        rgb_path="data/supervised/Agriculture-Vision-2021/train/images/rgb/ZP9VV1BTQ_1001-10861-1513-11373.jpg",
        nir_path="data/supervised/Agriculture-Vision-2021/train/images/nir/ZP9VV1BTQ_1001-10861-1513-11373.jpg",
        model_path="checkpoints/best_model_20250605_002445.pth",
        class_names=class_names,
        threshold=0.5
    )