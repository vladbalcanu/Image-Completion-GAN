# flake8: noqa
import torch
import torch.nn as nn

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(255.0).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, original_image, completed_image):
        mean_squared_error = torch.mean((original_image.float() - completed_image.float()) ** 2)
        if mean_squared_error == 0:
            return torch.tensor(0)
        return self.max_val - 10 * torch.log(mean_squared_error) / self.base10