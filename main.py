import yaml

from dataloader import get_dataloader
from metrics import frequency_weighted_intersection_over_union
from model import SegmentationUNet
from trainer import Trainer
from visualizer import Visualizer


def load_config():
    with open("config.yaml", "rt") as file:
        return yaml.safe_load(file.read())


if __name__ == "__main__":
    config = load_config()
    dataset_path = config["dataset_path"]
    train_dataloader = get_dataloader(dataset_path, "train")
    validation_dataloader = get_dataloader(dataset_path, "validation")
    test_dataloader = get_dataloader(dataset_path, "test")
    Visualizer.visualise_dataloader_samples(test_dataloader, 3, title="Some samples")

    model = SegmentationUNet(num_channels=3)
    trainer = Trainer(
        **config["trainer"],
        model=model,
        accuracy_function=frequency_weighted_intersection_over_union,
        is_model_binary=True
    )
    trainer.train(train_dataloader, validation_dataloader)
