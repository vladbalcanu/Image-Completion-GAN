# flake8: noqa
import torch.nn as nn
import torch
from helper_blocks.residual_block import ResidualBlock
from helper_blocks.self_attention_block import SelfAttentionBlock

class EdgeGenerator(nn.Module):
    def __init__(self, residual_blocks=8):
        super(EdgeGenerator, self).__init__()
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                with torch.no_grad():
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        self.encoder = nn.Sequential(
                                nn.ReflectionPad2d(3),
                                nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0)),
                                nn.InstanceNorm2d(64, track_running_stats=False),
                                nn.ReLU(True),

                                nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)),
                                nn.InstanceNorm2d(128, track_running_stats=False),
                                nn.ReLU(True),

                                nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
                                nn.InstanceNorm2d(256, track_running_stats=False),
                                nn.ReLU(True))

        self.self_attention = SelfAttentionBlock(256)

        blocks = []
        for _ in range(residual_blocks):
            block = ResidualBlock(256)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
                            nn.utils.parametrizations.spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)),
                            nn.InstanceNorm2d(128, track_running_stats=False),
                            nn.ReLU(True),

                            nn.utils.parametrizations.spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)),
                            nn.InstanceNorm2d(64, track_running_stats=False),
                            nn.ReLU(True),

                            nn.ReflectionPad2d(3),
                            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),)
        self.apply(init_func)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.self_attention(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x