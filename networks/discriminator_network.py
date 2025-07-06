# flake8: noqa
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                with torch.no_grad():
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        self.conv1 = nn.Sequential(nn.utils.parametrizations.spectral_norm(
                                nn.Conv2d(in_channels=in_channels, out_channels=64,kernel_size= 4, stride=2, padding=1, bias=False)),
                                nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.utils.parametrizations.spectral_norm(
                                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)),
                                nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.utils.parametrizations.spectral_norm(
                                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)),
                                nn.LeakyReLU(0.2, inplace=True),)

        self.conv4 = nn.Sequential(nn.utils.parametrizations.spectral_norm(
                                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False)),
                                nn.LeakyReLU(0.2, inplace=True),)

        self.conv5 = nn.Sequential(nn.utils.parametrizations.spectral_norm(
                                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False)),)

        self.apply(init_func)

    def forward(self, x):
        features_1 = self.conv1(x)
        features_2 = self.conv2(features_1)
        features_3 = self.conv3(features_2)
        features_4 = self.conv4(features_3)
        features_5 = self.conv5(features_4)
        prediction = torch.sigmoid(features_5)
        return prediction, [features_1, features_2, features_3, features_4, features_5]
