import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


class Visualizer:
    @classmethod
    def visualise_some_samples(cls, dataloader: DataLoader):
        batch_size = dataloader.batch_size
        print(batch_size)
        batches_to_plot = 3
        rows, cols = batch_size, batches_to_plot * 2
        plt.title("Some samples")
        plt.axis("off")
        for i_batch, sample_batched in enumerate(dataloader):
            imgs = sample_batched[0]
            segs = sample_batched[1]

            for i in range(0, batch_size):
                plt.subplot(rows, cols, cols * i + i_batch * 2 + 1)
                plt.axis("off")
                plt.imshow(imgs[i].numpy())

                plt.subplot(rows, cols, cols * i + i_batch * 2 + 2)
                plt.axis("off")
                plt.imshow(segs[i].numpy(), cmap="gray")

            if i_batch == batches_to_plot - 1:
                break

        plt.show()
