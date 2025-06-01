import os

import torch
import torch.nn as nn
from tqdm import tqdm

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from loaders.dataset import AgriVisionDataset


def train_one_epoch(model, loss_fn, loader, optimizer):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader):
        images, masks = images.cuda(), masks.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loss_fn, loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()
    return running_loss / len(loader)


def main():
    train_transform = A.Compose(
        [
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5,) * 4, std=(0.5,) * 4),  # 4-channel normalization
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [A.Resize(512, 512), A.Normalize(mean=(0.5,) * 4, std=(0.5,) * 4), ToTensorV2()]
    )

    train_dataset = AgriVisionDataset(
        os.path.join("data", "supervised", "Agriculture-Vision-2021", "train"),
        transform=train_transform,
    )
    val_dataset = AgriVisionDataset(
        os.path.join("data", "supervised", "Agriculture-Vision-2021", "train"),
        transform=val_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=4,  # RGB + NIR
        classes=8,  # number of classes
    ).to("cuda")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer)
        val_loss = validate(model, loss_fn, val_loader)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
