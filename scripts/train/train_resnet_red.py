import sys
import os

from diff_cf_ir.file_utils import dump_args


sys.path.append("/home/tha/master-thesis-xai/diff_cf_ir")
sys.path.append(os.getcwd())

import torch
from diff_cf_ir.train import setup_trainer
from diff_cf_ir.red_dataset import RedDataModule
from diff_cf_ir.models import ResNetRegression
import argparse
import json


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train ResNet for red values")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="/data/imdb-clean/imdb-clean-1024-cropped",
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="red",
        help="Name for the output folder",
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Size to resize images to"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
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

    model = ResNetRegression()

    args.seed = 1 if args.oracle else 0
    trainer = setup_trainer(name=args.name, seed=args.seed)
    data_module = RedDataModule(
        args.folder_path,
        img_size=args.image_size,
        batch_size=args.batch_size,
        trainer=trainer,
    )

    print("Running with the following arguments:")
    print(args)
    dump_args(args, trainer.logger.log_dir)

    trainer.fit(model=model, datamodule=data_module)
