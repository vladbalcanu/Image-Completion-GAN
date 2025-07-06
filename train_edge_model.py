# flake8: noqa
import torch
from utils.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import Places365
from torch.utils.data import DataLoader, Subset
from models.edge_model import EdgeModel
from metrics.precision_and_recall import PrecisionAndRecall

def main():
    transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor(),])
    places365_dataset = Places365(root='./places365', split='train-standard', small=True, transform=transform, download=True)
    train_dataset_fraction = 0.01
    train_dataset_size = int(train_dataset_fraction * len(places365_dataset))
    start_position = (len(places365_dataset) - train_dataset_size) // 2
    end_position = start_position + train_dataset_size
    train_subset = Subset(places365_dataset, list(range(start_position, end_position)))


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_model = EdgeModel().to(DEVICE)
    edge_model.load("final_phase")

    epoch = 0
    precision_and_recall = PrecisionAndRecall().to(DEVICE)
    train_dataset = Dataset(train_subset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=0,
        drop_last=True,
        shuffle=True
    )

    while epoch < 100:
        print('\n\nTraining epoch: %d' % epoch)

        for batch in train_loader:
            edge_model.train()

            images, images_gray, edges, masks = batch

            images = images.to(DEVICE)
            images_gray = images_gray.to(DEVICE)
            edges = edges.to(DEVICE)
            masks = masks.to(DEVICE)

            generated_edges, generator_loss, discriminator_loss, logs = edge_model.process(images_gray, edges, masks)

            precision, recall = precision_and_recall(edges * masks, generated_edges * masks)
            logs.append(('precision', precision.item()))
            logs.append(('recall', recall.item()))

            edge_model.backward(generator_loss, discriminator_loss)


            if edge_model.iteration % 100 == 0:
                logs.append(('iteration', edge_model.iteration))
                logs = [
                    ("epoch", epoch),
                    ("iteration", edge_model.iteration),
                ] + logs
                print(logs)

            if edge_model.iteration % 1000 == 0:
                edge_model.save(str(edge_model.iteration))

        epoch += 1
        
if __name__ == "__main__":
    main()
