from diff_cf_ir.file_utils import dump_args

import torch

from diff_cf_ir.healthcare_datasets import RetinaMNISTDataModule
from diff_cf_ir.models import ResNetRegression
from diff_cf_ir.train import setup_trainer
import argparse


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train ResNet on Retina MNIST.")
    parser.add_argument(
        "--name",
        type=str,
        default="retinaMNIST",
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

    torch.set_float32_matmul_precision("medium")

    resnet_type = "resnet18"
    model = ResNetRegression(resnet_type, small_images=True)

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    args.seed = 1 if args.oracle else 0
    trainer = setup_trainer(args.name, seed=args.seed)

    data_module = RetinaMNISTDataModule(args.batch_size)

    print("Running with the following arguments:")
    print(args)

    dump_args(args, trainer.logger.log_dir)
    trainer.fit(model=model, datamodule=data_module)
