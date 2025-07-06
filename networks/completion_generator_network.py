# flake8: noqa
import torch.nn as nn
import torch
from helper_blocks.residual_block import ResidualBlock
from helper_blocks.fusion_block import FusionBlock

class ImageCompletionGenerator(nn.Module):
    def __init__(self, residual_blocks=8):
        super(ImageCompletionGenerator, self).__init__()
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                with torch.no_grad():
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        self.encoder = nn.Sequential(
                            nn.ReflectionPad2d(3),
                            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
                            nn.InstanceNorm2d(64, track_running_stats=False),
                            nn.ReLU(True),

                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                            nn.InstanceNorm2d(128, track_running_stats=False),
                            nn.ReLU(True),

                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                            nn.InstanceNorm2d(256, track_running_stats=False),
                            nn.ReLU(True))

        blocks = []
        for _ in range(residual_blocks):
            block = ResidualBlock(256)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0)
        )

        self.fusion1 = FusionBlock(c_feat=128)
        self.fusion2 = FusionBlock(c_feat=64)

        self.proj1 = nn.Conv2d(3, 128, kernel_size=1)
        self.proj2 = nn.Conv2d(3, 64, kernel_size=1)

        self.apply(init_func)

    def forward(self, x):
        masked_image = x[:, :3, :, :]
        masked_image_and_edge_mask = x[:, :4, :, :] 
        mask = x[:, 4:5, :, :]

        x = self.encoder(masked_image_and_edge_mask)
        x = self.middle(x)

        x = self.upsample1(x)
        alpha_mask_1 = self.fusion1(x, torch.nn.functional.interpolate(mask, size=x.shape[2:], mode='nearest'))
        masked_image_projection_1 = torch.nn.functional.interpolate(self.proj1(torch.nn.functional.interpolate(masked_image, size=x.shape[2:], mode='bilinear', align_corners=False)), size=x.shape[2:])
        x = alpha_mask_1 * x + (1 - alpha_mask_1) * masked_image_projection_1

        x = self.upsample2(x)
        alpha_mask_2 = self.fusion2(x, mask)
        masked_image_projection_2 = self.proj2(masked_image)
        x = alpha_mask_2 * x + (1 - alpha_mask_2) * masked_image_projection_2

        x = self.final_conv(x)
        x = (torch.tanh(x) + 1) / 2

        return x