import numpy as np
import torch
import torch.nn as nn
import tqdm

from torch.utils.data import DataLoader

from visualizer import Visualizer


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        learning_rate: int,
        accuracy_function: callable,
        is_model_binary: bool,
    ):
        self.model = model
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(
            model.parameters(), learning_rate, weight_decay=1e-4
        )
        self.is_binary = is_model_binary
        if is_model_binary:
            self.loss = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0, 4.0]))
        else:
            self.loss = nn.CrossEntropyLoss()
        self.accuracy = accuracy_function

    def _train_step(self, dataloader: DataLoader, requires_grad: bool, desc: str):
        if requires_grad:
            self.model.train()
        else:
            torch.set_grad_enabled(False)
            self.model.eval()

        step_loss = 0.0
        step_accuracy = 0.0
        tqdm_dataloader = tqdm.tqdm(dataloader, desc=desc)
        for i, batch in enumerate(tqdm_dataloader, start=1):
            images, masks = batch
            if requires_grad:
                self.optimizer.zero_grad()

            predictions = self.model(images)

            if self.is_binary:
                flatten_preds = torch.flatten(predictions, start_dim=1)
            else:
                flatten_preds = torch.flatten(predictions, start_dim=2)

            loss = self.loss(flatten_preds, torch.flatten(masks, start_dim=1))

            if requires_grad:
                loss.backward()
                self.optimizer.step()
            step_loss += loss.item()
            step_accuracy += self._compute_metrics(predictions, masks)
            tqdm_dataloader.set_description(
                desc + f" | Loss: {step_loss / i} | Accuracy: {step_accuracy / i}"
            )
        step_loss /= len(dataloader)
        step_accuracy /= len(dataloader)
        if not requires_grad:
            torch.set_grad_enabled(True)
        return step_loss, step_accuracy

    def _compute_metrics(self, predictions: torch.Tensor, masks: torch.Tensor):
        masks = masks.numpy()
        if self.is_binary:
            predictions = (predictions >= 0.5).float().numpy()
        else:
            predictions = np.argmax(predictions.detach().numpy(), axis=1)
        return np.average(
            [self.accuracy(mask, pred) for mask, pred in zip(masks, predictions)]
        )

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
        loss_history = {"train": [], "validation": []}
        accuracy_history = {"train": [], "validation": []}
        for epoch in range(self.epochs):
            Visualizer.visualize_model_predictions(
                self.model,
                train_dataloader,
                3,
                title=f"Epoch {epoch}/{self.epochs}",
                is_binary=self.is_binary,
            )
            train_loss, train_accuracy = self._train_step(
                train_dataloader,
                requires_grad=True,
                desc=f"Train Epoch {epoch}/{self.epochs}",
            )
            loss_history["train"].append(train_loss)
            accuracy_history["train"].append(train_accuracy)
            val_loss, val_accuracy = self._train_step(
                val_dataloader,
                requires_grad=False,
                desc=f"Validation Epoch {epoch}/{self.epochs}",
            )
            loss_history["validation"].append(val_loss)
            accuracy_history["validation"].append(val_accuracy)

        Visualizer.visualize_model_predictions(
            self.model, train_dataloader, 3, title=f"Last Epoch"
        )
        return loss_history, accuracy_history
