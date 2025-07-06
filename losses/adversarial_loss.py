# flake8: noqa
import torch.nn as nn
import torch

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.criterion = nn.BCELoss()

    def __call__(self, outputs, is_real):
        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        loss = self.criterion(outputs, labels)
        return loss
        