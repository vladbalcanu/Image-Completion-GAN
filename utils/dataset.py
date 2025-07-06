# flake8: noqa
import torch
import numpy as np
import random
import torchvision
from PIL import Image
import torchvision.transforms.functional
from skimage.feature import canny
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super(Dataset, self).__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_item(self, index):
        image_tensor, _ = self.data[index]

        image_numpy = image_tensor.permute(1, 2, 0).numpy() * 255
        image_numpy = image_numpy.astype(np.uint8)
        image_rgb = image_numpy

        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        image_height, image_width = image_rgb.shape[:2]
        mask = self.create_mask(image_height, image_width, 0.10, 0.50)

        edges = canny(image_gray, sigma=2).astype(np.float64)

        if np.random.binomial(1, 0.5) > 0:
            image_rgb = image_rgb[:, ::-1, ...]
            image_gray = image_gray[:, ::-1, ...]
            edges = edges[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        original_image_tensor = self.to_tensor(image_rgb)
        image_gray_tensor = self.to_tensor(image_gray)
        edges_tensor = self.to_tensor(edges)
        mask_tensor = self.to_tensor(mask)

        return original_image_tensor, image_gray_tensor, edges_tensor, mask_tensor

    def create_mask(self, width, height, min_area = 0.01, max_area = 0.01):
        mask = np.zeros((height, width))
        min_area_pct = min_area
        max_area_pct = max_area

        total_pixels = width * height
        min_area = int(min_area_pct * total_pixels)
        max_area = int(max_area_pct * total_pixels)

        current_area = 0
        mask[:] = 0

        while current_area <= min_area:
            for _ in range(random.randint(1, 3)):
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2 = x1 + random.randint(-width // 4, width // 4)
                y2 = y1 + random.randint(-height // 4, height // 4)
                thickness = random.randint(10, 20)
                cv2.line(mask, (x1, y1), (x2, y2), 1, thickness)
            current_area = cv2.countNonZero(mask)
            if current_area > max_area:
                mask[:] = 0
                current_area = 0
        return mask

    def to_tensor(self, image):
        if len(image.shape) == 2:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray(image.astype(np.uint8))
        return torchvision.transforms.functional.to_tensor(image).float()