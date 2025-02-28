import argparse

import torch
import torchvision.transforms as T

from diff_cf_ir.basf_dataset import BASFDataModule
from diff_cf_ir.file_utils import assert_paths_exist
from diff_cf_ir.models import ResNetRegression
from diff_cf_ir.train import setup_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet on the BASF Dataset")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="/data/basf_resize512",
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="basf",
        help="Name for the output folder",
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="Size to resize images to"
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
    assert_paths_exist([args.folder_path])

    torch.set_float32_matmul_precision("medium")

    model = ResNetRegression("resnet152")

    args.seed = 1 if args.oracle else 0
    trainer = setup_trainer(args.name, seed=args.seed)

    transforms = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(360),
            T.ToTensor(),
        ]
    )

    data_module = BASFDataModule(
        args.folder_path,
        "combined",
        batch_size=args.batch_size,
        transform=transforms,
    )
    trainer.fit(model=model, datamodule=data_module)
