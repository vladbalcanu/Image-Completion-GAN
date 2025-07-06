# flake8: noqa
import torch
from torch.utils.data import DataLoader
from utils.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import Places365
from torch.utils.data import Subset
from models.edge_model import EdgeModel
from models.image_completion_model import CompletionModel
from metrics.precision_and_recall import PrecisionAndRecall
from metrics.psnr import PSNR

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
    edge_model = EdgeModel().to(DEVICE)
    completion_model = CompletionModel().to(DEVICE)
    edge_model.load("final_phase")
    completion_model.load("final_phase")

    psnr = PSNR().to(DEVICE)
    precision_and_recall = PrecisionAndRecall().to(DEVICE)
    train_dataset = Dataset(train_subset)

    epoch = 0
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
            edge_model.train()
            completion_model.train()

            images, images_gray, edges, masks = batch

            images = images.to(DEVICE)
            images_gray = images_gray.to(DEVICE)
            edges = edges.to(DEVICE)
            masks = masks.to(DEVICE)

            generated_edges, edge_generator_loss, edge_discriminator_loss, edge_model_loss_logs = edge_model.process(images_gray, edges, masks)
            generated_edges = generated_edges * masks + edges * (1 - masks)
            completed_images, completion_generator_loss, completion_discriminator_loss, completion_model_loss_logs = completion_model.process(images, generated_edges, masks)
            completed_images_merged = (completed_images * masks) + (images * (1 - masks))

            psnr_values = psnr(postprocess(images), postprocess(completed_images_merged))
            mae = (torch.sum(torch.abs(images - completed_images_merged)) / torch.sum(images)).float()
            precision, recall = precision_and_recall(edges * masks, generated_edges * masks)
            
            edge_model_loss_logs.append(('precision', precision.item()))
            edge_model_loss_logs.append(('recall', recall.item()))
            completion_model_loss_logs.append(('psnr', psnr_values.item()))
            completion_model_loss_logs.append(('mae', mae.item()))
            logs = edge_model_loss_logs + completion_model_loss_logs

            completion_model.backward(completion_generator_loss, completion_discriminator_loss)
            edge_model.backward(edge_generator_loss, edge_discriminator_loss)

            if edge_model.iteration % 10 == 0:
                logs = [
                ("epoch", epoch),
                ("iteration_completion_model", completion_model.iteration),
                ("iteration_edge_model", edge_model.iteration),
                ] + logs
                print(logs)

            if edge_model.iteration % 50 == 0:
                edge_model.save(edge_model.iteration)
                completion_model.save(completion_model.iteration)

        epoch += 1
        

if __name__ == "__main__":
    main()
