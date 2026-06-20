"""
Lightweight CNN for eye open/closed binary classification.
Uses MobileNetV2 backbone (pretrained on ImageNet) with a custom head.
Input: 64x64 RGB eye-region crop.
Output: [closed_prob, open_prob]
"""

import torch
import torch.nn as nn
from torchvision import models


class EyeStateModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Replace classifier
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)


def load_model(checkpoint_path, device="cpu"):
    model = EyeStateModel(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, state.get("threshold", 0.5)
