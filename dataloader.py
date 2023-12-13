import os

import cv2

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LfwDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        image_transform: callable = None,
        mask_transform: callable = None,
    ):
        images_location = os.path.join(path, "images")
        masks_location = os.path.join(path, "masks")
        image_path_pairs = []
        with open(os.path.join(path, f"parts_{split}.txt"), "r") as f:
            for line in f.readlines():
                name, index = line.strip().split(" ")
                index = "0" * (4 - len(index)) + index
                image_path_pairs.append(
                    (
                        os.path.join(images_location, name, f"{name}_{index}.jpg"),
                        os.path.join(masks_location, f"{name}_{index}.ppm"),
                    )
                )

        self._image_path_pairs = image_path_pairs
        self._image_transform = image_transform
        self._mask_transform = mask_transform

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        paths = self._image_path_pairs[idx]
        image = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
        if self._image_transform is not None:
            image = self._image_transform(image)

        mask = cv2.cvtColor(cv2.imread(paths[1]), cv2.COLOR_BGR2RGB)
        if self._mask_transform is not None:
            mask = self._mask_transform(mask)
        return image, mask

    def __len__(self):
        return len(self._image_path_pairs)


def get_dataloader(path: str, split: str, **kwargs) -> DataLoader:
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((240, 240), antialias=True),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Lambda(lambda mask: np.argmax(mask, axis=2)),
            np.vectorize(lambda x: 1 if x == 0 else 0),
            transforms.ToTensor(),
            transforms.Resize((240, 240), antialias=True),
            transforms.Lambda(lambda x: x.float()),
        ]
    )
    dataset = LfwDataset(path, split, image_transform, mask_transform)
    return DataLoader(
        dataset,
        **kwargs,
    )
