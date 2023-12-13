import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader


class Visualizer:
    @classmethod
    def _plotable_image(cls, img: torch.Tensor, is_normalized: bool = False):
        if len(img.shape) != 3:
            return img
        img = np.transpose(img.numpy(), (1, 2, 0))
        if is_normalized:
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
        return img

    @classmethod
    def visualise_dataloader_samples(
        cls, dataloader: DataLoader, batches_to_plot: int, title: str = None
    ):
        batch_size = dataloader.batch_size
        rows, cols = batch_size, batches_to_plot * 2
        if title is not None:
            plt.title(title)
        plt.axis("off")
        for i_batch, batch in enumerate(dataloader):
            imgs, masks = batch
            for i in range(0, batch_size):
                plt.subplot(rows, cols, cols * i + i_batch * 2 + 1)
                plt.axis("off")
                plt.imshow(cls._plotable_image(imgs[i], is_normalized=True))

                plt.subplot(rows, cols, cols * i + i_batch * 2 + 2)
                plt.axis("off")
                plt.imshow(cls._plotable_image(masks[i]), cmap="binary")

            if i_batch == batches_to_plot - 1:
                break
        plt.show()

    @classmethod
    def visualize_model_predictions(
        cls,
        model: nn.Module,
        dataloader: DataLoader,
        examples_count: int,
        title: str = None,
        is_binary: bool = False,
    ):
        model.eval()
        shown_so_far = 0
        batch_size = dataloader.batch_size
        with torch.no_grad():
            for batch in dataloader:
                imgs, masks = batch
                preds = model(imgs)
                if not is_binary:
                    preds = torch.argmax(preds, dim=1)
                else:
                    preds = (preds >= 0.5).float()
                for i in range(batch_size):
                    # plot
                    plt.subplot(examples_count, 3, 3 * shown_so_far + 1)
                    if shown_so_far == 0:
                        plt.title("Image")
                    plt.imshow(cls._plotable_image(imgs[i], is_normalized=True))
                    plt.axis("off")

                    plt.subplot(examples_count, 3, 3 * shown_so_far + 2)
                    if shown_so_far == 0:
                        plt.title(
                            "\n".join([title, "GT Mask"])
                            if title is not None
                            else "GT Mask"
                        )
                    plt.imshow(cls._plotable_image(masks[i]), cmap="binary")
                    plt.axis("off")

                    plt.subplot(examples_count, 3, 3 * shown_so_far + 3)
                    if shown_so_far == 0:
                        plt.title("Pred Mask")
                    print(preds[i].shape)
                    plt.imshow(cls._plotable_image(preds[i]), cmap="binary")
                    plt.axis("off")

                    shown_so_far += 1
                    if shown_so_far == examples_count:
                        plt.show()
                        return
