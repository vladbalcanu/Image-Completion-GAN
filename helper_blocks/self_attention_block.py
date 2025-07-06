# flake8: noqa
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query_vector = self.query_conv(x).view(batch_size, -1, height * width)
        key_vector = self.key_conv(x).view(batch_size, -1, height * width)
        raw_attention_scores = torch.bmm(query_vector.permute(0, 2, 1), key_vector)
        attention_score_matrix = F.softmax(raw_attention_scores, dim=-1)
        value_vector = self.value_conv(x).view(batch_size, -1, height * width)

        out = torch.bmm(value_vector, attention_score_matrix.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out