import torchvision.models as models
import torch.nn as nn


def get_resnet50_encoder(input_channels=3):
    model = models.resnet50(pretrained=True)

    if input_channels != 3:
        # Replace first conv layer to accept more channels (e.g., 4)
        old_weights = model.conv1.weight.data
        model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            model.conv1.weight[:, :3] = old_weights  # Copy RGB weights
            if input_channels > 3:
                model.conv1.weight[:, 3:] = old_weights[
                    :, :1
                ]  # Initialize extra channel (NIR)

    return model
