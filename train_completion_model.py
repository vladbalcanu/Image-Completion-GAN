# flake8: noqa
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import Places365
from torch.utils.data import Subset
from models.image_completion_model import CompletionModel
from metrics.psnr import PSNR
import os
from datetime import datetime

def postprocess(image):
    image = image * 255.0
    image = image.permute(0, 2, 3, 1)
    return image.int()

def main():
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])
    places365_dataset = Places365(root='./places365', split='train-standard', small=True, transform=transform, download=True)
    train_dataset_fraction = 0.01
    train_dataset_size = int(train_dataset_fraction * len(places365_dataset))
    start_position = (len(places365_dataset) - train_dataset_size) // 2
    end_position = start_position + train_dataset_size
    train_subset = Subset(places365_dataset, list(range(start_position, end_position)))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completion_model = CompletionModel().to(DEVICE)
    completion_model.load(111000)

    epoch = 0
    psnr = PSNR().to(DEVICE)
    train_dataset = Dataset(train_subset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=0,
        drop_last=True,
        shuffle=True
    )

    while epoch < 100:
        print('\nTraining epoch: %d' % epoch)

        for batch in train_loader:
            completion_model.train()

            images, images_gray, edges, masks = batch

            images = images.to(DEVICE)
            images_gray = images_gray.to(DEVICE)
            edges = edges.to(DEVICE)
            masks = masks.to(DEVICE)

            completed_images, generator_loss, discriminator_loss, logs = completion_model.process(images, edges, masks)
            completed_images_merged = (completed_images * masks) + (images * (1 - masks))

            psnr_values = psnr(postprocess(images), postprocess(completed_images_merged))
            mae = (torch.sum(torch.abs(images - completed_images_merged)) / torch.sum(images)).float()
            logs.append(('psnr', psnr_values.item()))
            logs.append(('mae', mae.item()))

            completion_model.backward(generator_loss, discriminator_loss)

            if completion_model.iteration % 100 == 1:
                logs.append(('iteration', completion_model.iteration))
                logs = [
                ("epoch", epoch),
                ("iter", completion_model.iteration),
                ] + logs
                print(logs)

            if completion_model.iteration % 500 == 0:
                completion_model.save(completion_model.iteration)

        epoch += 1
        
if __name__ == "__main__":
    main()
