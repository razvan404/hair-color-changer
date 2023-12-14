import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataloader import get_dataloader


def get_dataloaders(
    dataset_path: str, batch_size: int
) -> (DataLoader, DataLoader, DataLoader):
    train_dataloader = get_dataloader(dataset_path, "train", batch_size, True)
    validation_dataloader = get_dataloader(
        dataset_path, "validation", batch_size, False
    )
    test_dataloader = get_dataloader(dataset_path, "test", batch_size, False)
    return train_dataloader, validation_dataloader, test_dataloader


def save_checkpoint(state, filename: str):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(filename: str, model: nn.Module):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(filename))


def binary_predictions(raw_preds: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(raw_preds) >= 0.5).float()


def multiclass_predictions(raw_preds: torch.Tensor) -> torch.Tensor:
    return torch.argmax(raw_preds, dim=1)