# flake8: noqa
import torch
from torch.utils.data import DataLoader
from utils.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import Places365
from torch.utils.data import Subset
from models.edge_model import EdgeModel
from metrics.precision_and_recall import PrecisionAndRecall

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

    edge_model = EdgeModel().to(DEVICE)
    edge_model.load("final_phase")
    edge_model.eval()

    precision_and_recall = PrecisionAndRecall().to(DEVICE)
    validation_dataset = Dataset(validation_subset)

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=8,
        num_workers=0,
        drop_last=True,
        shuffle=True
    )

    avg_precision = 0.00000
    avg_recall = 0.00000

    with torch.no_grad():
        for batch in validation_loader:
            images, images_gray, edges, masks = [x.to(DEVICE) for x in batch]

            generated_edges, generator_loss, discriminator_loss, edge_model_loss_logs = edge_model.process(images_gray, edges, masks)
            precision, recall = precision_and_recall(edges * masks, generated_edges * masks)
            avg_precision += precision.item()
            avg_recall += recall.item()

    avg_precision = avg_precision / 1250
    avg_recall = avg_recall / 1250

    print(f"Final Precision over 10000 images: {avg_precision:.4f}")
    print(f"Final Recall over 10000 images:    {avg_recall:.4f}")

if __name__ == "__main__":
    main()
