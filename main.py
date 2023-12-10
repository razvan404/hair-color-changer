import yaml

from dataloader import get_dataloader
from visualizer import Visualizer


def load_config():
    with open("config.yaml", "rt") as file:
        return yaml.safe_load(file.read())


if __name__ == "__main__":
    config = load_config()
    dataset_path = config["dataset"]["path"]
    train_dataloader = get_dataloader(
        dataset_path, "train", **config["train_dataloader_options"]
    )
    test_dataloader = get_dataloader(
        dataset_path, "test", **config["test_dataloader_options"]
    )
    Visualizer.visualise_some_samples(test_dataloader)
