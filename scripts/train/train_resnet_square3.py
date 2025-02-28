import sys
import os

from thesis_utils.file_utils import deterministic_run, dump_args
from thesis_utils.train import setup_trainer

sys.path.append("/home/tha/master-thesis-xai/thesis_utils")
sys.path.append(os.getcwd())

from lightning import seed_everything

import torch
import torchvision.transforms as transforms

from thesis_utils.squares_dataset import SquaresDataModule
from thesis_utils.models import ResNetRegression
import argparse


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train ResNet on the Square3 Dataset")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="/data/square3",
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="square3",
        help="Name for the output folder",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Set seed to be for oracle model (1).",
    )

    args = parser.parse_args()

    # Check directories
    if not os.path.exists(args.folder_path):
        raise FileNotFoundError(f"Folder {args.folder_path} does not exist")

    torch.set_float32_matmul_precision("medium")

    # Seed everything for reproducibility
    args.seed = 1 if args.oracle else 0
    deterministic_run(seed=args.seed)

    model = ResNetRegression(type="resnet18", small_images=True)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )

    data_module = SquaresDataModule(
        args.folder_path, batch_size=args.batch_size, transform=transform
    )

    trainer = setup_trainer(name=args.name, seed=args.seed)
    print("Running with the following arguments:")
    print(args)

    dump_args(args, trainer.logger.log_dir)
    trainer.fit(model=model, datamodule=data_module)
