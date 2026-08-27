import torch
import torch.nn as nn
from torchvision import models


def load_model(model_path, device="cpu"):
    """
    Load the trained ResNet50 model from .pth file
    """
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification

    # Load weights safely (works on both CPU and GPU)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model