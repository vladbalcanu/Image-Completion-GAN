# flake8: noqa
import torch.nn as nn
import torch
from losses.vgg19 import VGG19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        perceptual_loss = 0.0
        perceptual_loss += self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        perceptual_loss += self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        perceptual_loss += self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        perceptual_loss += self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        perceptual_loss += self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
        return perceptual_loss
