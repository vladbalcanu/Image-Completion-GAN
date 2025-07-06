# flake8: noqa
import torch
import torch.nn as nn

class PrecisionAndRecall(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, original_edges, generated_edges):
        original_edges_filtered = (original_edges > 0.5)
        generated_edges_filtered = (generated_edges > 0.5)

        true_positive = ((generated_edges_filtered == original_edges_filtered) & original_edges_filtered).float()
        recall = true_positive.sum() / (original_edges_filtered.sum().float() + 1e-8)
        precision = true_positive.sum() / (generated_edges_filtered.sum().float() + 1e-8)

        return precision, recall