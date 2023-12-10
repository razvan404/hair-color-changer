import os
from typing import Optional, Callable, List

import cv2

import numpy as np

from torch.utils.data import Dataset, DataLoader


class LfwDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = 'train',
        transforms: Optional[List[Callable]] = None,
    ):
        images_location = os.path.join(path, 'images')
        masks_location = os.path.join(path, 'masks')
        image_path_pairs = []
        with open(os.path.join(path, f"parts_{split}.txt"), 'r') as f:
            for line in f.readlines():
                name, index = line[:-1].split(" ")
                index = "0" * (4 - len(index)) + index
                image_path_pairs.append((
                    os.path.join(images_location, name, f"{name}_{index}.jpg"),
                    os.path.join(masks_location, f"{name}_{index}.ppm"),
                ))

        self._image_path_pairs = image_path_pairs
        self._transforms = transforms

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        paths = self._image_path_pairs[idx]
        image = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(paths[1]), cv2.COLOR_BGR2RGB)
        return image, mask

    def __len__(self):
        return len(self._image_path_pairs)


def get_dataloader(path: str, split: str, **kwargs) -> DataLoader:
    dataset = LfwDataset(path, split)
    return DataLoader(
        dataset,
        **kwargs,
    )
