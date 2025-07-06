# flake8: noqa
import torch
from torch.utils.data import DataLoader
from utils.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import Places365
from torch.utils.data import Subset
from models.image_completion_model import CompletionModel
from metrics.psnr import PSNR
from skimage.metrics import structural_similarity
from models.edge_model import EdgeModel
import cv2
import numpy as np


def postprocess(image):
    image = image * 255.0
    image = image.permute(0, 2, 3, 1)
    return image.int()

def calculate_ssim_for_batch(original_images, predicted_images, batch_size):
    avg_ssim = 0
    for i in range(0, batch_size):
        original_image = original_images[i].permute(1, 2, 0).cpu().numpy()
        predicted_image = predicted_images[i].squeeze().cpu().numpy() / 255.0
        ssim = structural_similarity(original_image, predicted_image, win_size=11, channel_axis=2, data_range=1.0) 
        avg_ssim += ssim
    avg_ssim = avg_ssim / batch_size
    return avg_ssim

def clean_generated_edges_batch(generated_edges):
    batch_size = generated_edges.shape[0]
    processed_edges = []
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = np.ones((3,3), np.uint8)

    for i in range(batch_size):
        edges_numpy = generated_edges[i, 0].cpu().numpy() * 255
        edges_numpy = edges_numpy.astype(np.uint8)

        _, normalized_edges = cv2.threshold(edges_numpy, 127, 255, cv2.THRESH_BINARY)
        dilated_edges = cv2.dilate(normalized_edges, kernel, iterations=1)
        eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

        closed_edges = eroded_edges.astype(np.float32) / 255.0
        closed_edges = torch.from_numpy(closed_edges).unsqueeze(0)
        processed_edges.append(closed_edges)

    cleaned_edges = torch.stack(processed_edges, dim=0)
    return cleaned_edges.to(DEVICE)   
     
def main():
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])
    places365_dataset= Places365(root='./places365', split='train-standard', small=True, transform=transform, download=True)

    evaluation_dataset_fraction = 0.01
    evaluation_dataset_size = int(evaluation_dataset_fraction * len(places365_dataset))
    middle_start_position = (len(places365_dataset) - evaluation_dataset_size) // 2
    middle_end_position = middle_start_position + evaluation_dataset_size
    validation_subset_half_size = 5000

    before_batch_start = max(middle_start_position - validation_subset_half_size, 0)
    before_batch_end = middle_start_position
    after_batch_start = middle_end_position
    after_batch_end = min(middle_end_position + validation_subset_half_size, len(places365_dataset))

    evaluation_batches_subsets_positions = list(range(before_batch_start, before_batch_end)) + list(range(after_batch_start, after_batch_end))
    validation_subset = Subset(places365_dataset, evaluation_batches_subsets_positions)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    completion_model = CompletionModel().to(DEVICE)
    completion_model.load("final_phase")
    edge_model = EdgeModel().to(DEVICE)
    edge_model.load("final_phase")
    edge_model.eval()
    completion_model.eval()

    evaluation_dataset = Dataset(validation_subset)
    psnr = PSNR().to(DEVICE)

    evualuation_loader = DataLoader(
        dataset=evaluation_dataset,
        batch_size=8,
        num_workers=0,
        drop_last=True,
        shuffle=True
    )

    avg_psnr = 0.00000
    avg_mae = 0.00000
    avg_mae_combined = 0.00000
    avg_ssim = 0.00000

    with torch.no_grad():
        for batch in evualuation_loader:
            images, images_gray, edges, masks = batch

            images = images.to(DEVICE)
            images_gray = images_gray.to(DEVICE)
            edges = edges.to(DEVICE)
            masks = masks.to(DEVICE)
            
            generated_edges, edge_generator_loss, edge_discriminator_loss, edge_model_loss_logs = edge_model.process(images_gray, edges, masks)
            generated_edges_cleaned = clean_generated_edges_batch(generated_edges)
            generated_edges_clean_merged = generated_edges_cleaned * masks + edges * (1 - masks)

            completed_images, completion_generator_loss, completion_discriminator_loss, completion_model_loss_logs = completion_model.process(images, generated_edges_clean_merged, masks)
            completed_images_merged = (completed_images * masks) + (images * (1 - masks))
            postprocessed_completed_images = postprocess(completed_images_merged)

            ssim_for_batch = calculate_ssim_for_batch(images, postprocessed_completed_images, 8)
            psnr_values = psnr(postprocess(images), postprocessed_completed_images)
            mae = (torch.sum(torch.abs(images - completed_images_merged)) / torch.sum(images)).float()
            mae_combined = (torch.sum(torch.abs(images - completed_images_merged)) / (torch.sum(torch.abs(images) + torch.abs(completed_images_merged)))).float()

            avg_ssim += ssim_for_batch
            avg_psnr += psnr_values.item()
            avg_mae += mae.item()
            avg_mae_combined += mae_combined.item()

    avg_psnr = avg_psnr / 1250
    avg_mae = avg_mae / 1250
    avg_ssim = avg_ssim / 1250
    avg_mae_combined = avg_mae_combined / 1250

    print(f"Final PSNR over 10000 images: {avg_psnr:.4f}")
    print(f"Final MAE over 10000 images:    {avg_mae:.4f}")
    print(f"Final MAE combined over 10000 images:    {avg_mae_combined:.4f}")
    print(f"Final SSIM over 10000 images:    {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
