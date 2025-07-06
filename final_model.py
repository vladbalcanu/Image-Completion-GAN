# flake8: noqa
import torch
from models.image_completion_model import CompletionModel
from models.edge_model import EdgeModel
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import canny
from PIL import Image
import torchvision

class FinalModel():
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_model = EdgeModel().to(self.DEVICE)
        self.completion_model = CompletionModel().to(self.DEVICE)
        self.edge_model.load("first_phase")
        self.completion_model.load("first_phase")

    def complete_image(self, image, input_mask, show_intermediate_results: bool = True):
        image_numpy = image.permute(1, 2, 0).numpy() * 255
        image_numpy = image_numpy.astype(np.uint8)
        gray_image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)
        edge = canny(gray_image, sigma=2).astype(np.float64)
        self.edge_model.eval()
        self.completion_model.eval()

        with torch.no_grad():
            gray_image_tensor = self.to_tensor(gray_image)
            edges_tensor = self.to_tensor(edge)
            mask_tensor = self.to_tensor(input_mask)

            original_image_tensor = image.unsqueeze(0).to(self.DEVICE)
            gray_image_tensor = gray_image_tensor.unsqueeze(0).to(self.DEVICE)
            edges_tensor = edges_tensor.unsqueeze(0).to(self.DEVICE)
            mask_tensor = mask_tensor.unsqueeze(0).to(self.DEVICE)

            generated_edges, edge_genenrator_loss, edge_discriminator_loss, edge_model_loss_logs = self.edge_model.process(gray_image_tensor, edges_tensor, mask_tensor)
            generated_edges_merged = self.clean_generated_edges_batch(generated_edges * mask_tensor)
            generated_edges_merged_postprocessed = self.postprocess(generated_edges_merged)
            generated_edges_clean = self.clean_generated_edges_batch(generated_edges)
            generated_edges = generated_edges_clean * mask_tensor + edges_tensor * (1 - mask_tensor)
            
            completed_images, completion_generator_loss, completion_discriminator_loss, completion_model_loss_logs = self.completion_model.process(original_image_tensor, generated_edges, mask_tensor)
            completed_images_merged = (completed_images * mask_tensor) + (original_image_tensor * (1 - mask_tensor))
            completed_images_merged_postprocessed = self.postprocess(completed_images_merged)
            final_image = completed_images_merged_postprocessed[0].squeeze().cpu().numpy()

            if show_intermediate_results:
                original_image = original_image_tensor[0].permute(1, 2, 0).cpu().numpy()
                original_edge = edges_tensor[0].squeeze().cpu().numpy()
                mask = mask_tensor[0].squeeze().cpu().numpy()
                predicted_edge = generated_edges_merged_postprocessed[0].squeeze().cpu().numpy()
                _, axes = plt.subplots(1, 5, figsize=(16, 4))
                axes[0].imshow(original_image)
                axes[0].set_title(f"Original Image")
                axes[0].axis('off')
                axes[1].imshow(original_edge, cmap='gray')
                axes[1].set_title("Original Edge")
                axes[1].axis('off')
                axes[2].imshow(mask, cmap='gray')
                axes[2].set_title("Mask")
                axes[2].axis('off')
                axes[3].imshow(predicted_edge, cmap='gray')
                axes[3].set_title("Predicted Edge")
                axes[3].axis('off')
                axes[4].imshow(final_image, cmap='gray')
                axes[4].set_title("Predicted Image")
                axes[4].axis('off')
                plt.tight_layout()
                plt.show()
            else:
                plt.imshow(final_image, cmap='gray')
                plt.title('Predicted Image')
                plt.axis('off')
                plt.show()

        
    def to_tensor(self, image):
        if len(image.shape) == 2:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray(image.astype(np.uint8))
        return torchvision.transforms.functional.to_tensor(image).float()
    
    def clean_generated_edges_batch(self, generated_edges):
        batch_size = generated_edges.shape[0]
        processed_edges = []
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
        return cleaned_edges.to(self.DEVICE)   

    def postprocess(self, image):
        image = image * 255.0
        image = image.permute(0, 2, 3, 1)
        return image.int() 


