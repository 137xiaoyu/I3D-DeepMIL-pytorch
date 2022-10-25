import torch
from torch import nn
from Inception_V1_I3D import InceptionI3D


class Model(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super().__init__()

        i3d = InceptionI3D()

        if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            i3d.load_state_dict(checkpoint)

        i3d.replace_logits(num_classes)

        i3d_child_modules = list(i3d.children())

        self.backbone = nn.Sequential(*i3d_child_modules[:-2])

        self.dropout = nn.Dropout(0.5)
        self.logits = i3d_child_modules[-1]

    def forward(self, x):
        features = self.backbone(x)  # b 3 16 224 224 -> # b 1024 1 1 1
        logits = self.logits(self.dropout(features))  # b c 1 1 1

        logits = logits.squeeze(-1).squeeze(-1).squeeze(-1)  # b c
        features = features.squeeze(-1).squeeze(-1).squeeze(-1)  # b 1024

        return logits, features
