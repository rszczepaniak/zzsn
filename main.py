import torch.nn as nn
from datasets.dataset import MultiClassDataset
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
from models.unet import UNet
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch
import random
import pickle
import json
from datetime import datetime


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 4
    out_channels = 9
    model = UNet(in_channels, out_channels)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00045)
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

    log = {
        "hyperparameters": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "optimizer": "Adam",
            "learning_rate": 0.00045,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)",
            "batch_size": 32,
            "num_epochs": 20,
        },
        "train_history": [],
        "final_results": {},
    }

    best_model_path = train(
        device, model, optimizer, scheduler, train_dataset, val_dataset, log
    )
    test(device, model, test_dataset, best_model_path, log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/training_log_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {log_path}")


def train(device, model, optimizer, scheduler, train_dataset, val_dataset, log):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_model_path = "checkpoints/best_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(30):
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
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                true_labels = labels.float()
                correct_pixels += (predicted == true_labels).sum().item()
                total_pixels += true_labels.numel()

        train_acc = correct_pixels / total_pixels
        avg_train_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/30]\nTrain Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}"
        )

        # Validate after each epoch
        val_acc, val_loss, val_f1, val_iou = validate(device, model, val_dataset)

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
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Multi-label prediction: sigmoid + threshold
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            true_labels = labels.float()

            # For computing flat pixel-level accuracy
            correct_pixels += (predicted == true_labels).sum().item()
            total_pixels += true_labels.numel()

            # Flatten for metric logging
            all_preds.append(predicted.cpu().numpy().astype(int).flatten())
            all_trues.append(true_labels.cpu().numpy().astype(int).flatten())

    acc = correct_pixels / total_pixels
    avg_loss = total_loss / len(val_loader)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    iou = jaccard_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, avg_loss, f1, iou


def test(device, model, test_dataset, model_path, log=None):
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            true_labels = labels.float()

            all_preds.append(predicted.cpu().numpy().flatten())
            all_trues.append(true_labels.cpu().numpy().flatten())

            correct_pixels += (predicted == true_labels).sum().item()
            total_pixels += true_labels.numel()

    acc = correct_pixels / total_pixels
    avg_loss = total_loss / len(test_loader)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    iou = jaccard_score(y_true, y_pred, average="macro", zero_division=0)

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
    main()
