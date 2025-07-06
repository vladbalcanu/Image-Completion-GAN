# flake8: noqa
import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    def __init__(self, c_feat):
        super(FusionBlock, self).__init__()
        self.alpha_predictor = nn.Sequential(
            nn.Conv2d(c_feat + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features, mask):
        x = torch.cat([features, mask], dim=1)
        alpha = self.alpha_predictor(x)
        return alpha
