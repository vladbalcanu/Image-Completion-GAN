# flake8: noqa
import torch.nn as nn
import torch
from losses.vgg19 import VGG19

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        batch_size, channels, height, width = x.size()
        flattened_input = x.view(batch_size, channels, width * height)
        flattened_input_transposed = flattened_input.transpose(1, 2)
        gram_matrix = flattened_input.bmm(flattened_input_transposed) / (height * width * channels)
        return gram_matrix
    
    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))
        return style_loss