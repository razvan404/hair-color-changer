import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from torch.utils.data import DataLoader

from core.config import load_config
from .utils import get_dataloaders, binary_predictions, load_checkpoint, save_checkpoint
from .metrics import frequency_weighted_intersection_over_union
from .model import SegmentationUNet
from .visualizer import Visualizer


def train_step(
    loader: DataLoader,
    model: nn.Module,
    optimizer,
    loss_function: callable,
    accuracy_function: callable,
    predictions_function: callable,
    scaler,
    device: str,
    desc: str = None,
):
    loop = tqdm.tqdm(loader, desc=desc)
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)

        # forward
        preds = model(images)
        loss = loss_function(preds, masks).cpu()

        # backward
        if optimizer is not None and scaler is not None:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # display metrics
        preds = predictions_function(preds.cpu().detach()).numpy()
        accuracy = accuracy_function(masks.cpu().numpy(), preds)
        loop.set_postfix(loss=loss.item(), accuracy=accuracy)


def train():
    config = load_config("segmentation")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment = config["experiment"]

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        config["dataset_path"], config["batch_size"]
    )
    Visualizer.visualise_dataloader_samples(
        test_dataloader,
        3,
        4,
        title="Some samples",
        save_path=os.path.join(config["plots_path"], "samples.png"),
    )

    model = SegmentationUNet(3, 1).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    accuracy_function = frequency_weighted_intersection_over_union
    predictions_function = binary_predictions
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()

    if config["load_model"]:
        load_checkpoint(config["save_model_path"], model)
    Visualizer.visualize_model_predictions(
        model,
        val_dataloader,
        4,
        predictions_function,
        device,
        f"Epoch 0/{config['epochs']}",
        save_path=os.path.join(config["plots_path"], "results_e0.png")
        if not config["load_model"]
        else None,
    )
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_step(
            train_dataloader,
            model,
            optimizer,
            loss_function,
            accuracy_function,
            predictions_function,
            scaler,
            device,
            desc=f"Train epoch {epoch}/{config['epochs']}",
        )
        model.eval()
        with torch.no_grad():
            train_step(
                val_dataloader,
                model,
                None,
                loss_function,
                accuracy_function,
                predictions_function,
                None,
                device,
                desc=f"Validation epoch {epoch}/{config['epochs']}",
            )
            Visualizer.visualize_model_predictions(
                model,
                val_dataloader,
                4,
                predictions_function,
                device,
                f"Epoch {epoch}/{config['epochs']}",
            )
        save_checkpoint(model.state_dict(), config["save_model_path"])


if __name__ == "__main__":
    train()
