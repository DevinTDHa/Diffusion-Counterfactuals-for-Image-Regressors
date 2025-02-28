import os
import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor, Normalize
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import lightning as L

import monai.transforms as T
import monai.data as data


# Helper classes from https://github.com/matanat/dae_counterfactual/blob/main/dataset.py#L17


class CropMaskByLabel(T.Transform):
    def __init__(
        self, mask_key="mask", label_key="label", label_lambda_func=lambda x: x
    ):
        super().__init__()
        self.mask_key = mask_key
        self.label_key = label_key
        self.label_lambda_func = label_lambda_func

    def __call__(self, data):
        d = dict(data)

        mask = d[self.mask_key]
        label = self.label_lambda_func(d[self.label_key])
        d[self.mask_key] = (mask == label).astype(mask.dtype)

        assert d[self.mask_key].sum(), "patient %d, label %d" % (d["patient"], label)

        return d


class AssertEmptyImaged(T.Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        d = dict(data)

        assert d["image"].size()[-1], "patient %d, label %d, %s, %s" % (
            d["patient"],
            d["ivd_label"],
            str(d["image"].size()),
            str(d["mask"].size()),
        )

        return d


class SPIDERDataset(VisionDataset):
    """Loads the SPIDER dataset.

    Taken from https://github.com/matanat/dae_counterfactual/blob/main/config.py

    The dataset consists of MRI images of the spine and masks of the intervertebral discs (IVD).
    The dataset also contains the Pfirrman grade of the IVDs.
    """

    def __init__(self, root: str, split: str = "training", mode="train"):
        self.root = root
        self.split = split
        self.mode = mode

        self.dataset = self.create_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def create_spider_files(self):
        split = self.split
        meta = pd.read_csv(os.path.join(self.root, "overview.csv"))
        labels = pd.read_csv(os.path.join(self.root, "radiological_gradings.csv"))

        meta_train = meta[
            (meta.subset == split) & (meta.new_file_name.str.endswith("t2"))
        ]

        img_path = os.path.join(self.root, "images/")
        mask_path = os.path.join(self.root, "masks/")

        files = list()
        for f in list(meta_train.new_file_name):
            rec = dict()
            rec["image"] = img_path + f + ".mha"
            rec["mask"] = mask_path + f + ".mha"
            rec["patient"] = int(f.split("_")[0])

            num_vertebrae = meta_train[meta_train.new_file_name == f][
                "num_vertebrae"
            ].item()
            for v in range(1, num_vertebrae + 1):
                rec["ivd_label"] = v
                match = labels[
                    (labels["Patient"] == rec["patient"]) & (labels["IVD label"] == v)
                ]
                if len(match["Pfirrman grade"]) != 1:
                    print("Missing grade, skipping IVD")
                    continue
                rec["pfirrman_grade"] = match["Pfirrman grade"].item() - 1
                files.append(rec.copy())

        # something wrong with this
        files = [f for f in files if not (f["patient"] == 256 and f["ivd_label"]) == 8]
        return files

    def get_transform(self):
        # Include some augmentations
        if self.mode == "train":
            prob = 0.5
            transforms = T.Compose(
                [
                    # each image has a single ivd label
                    T.LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
                    T.Orientationd(keys=["image", "mask"], axcodes="RAS"),
                    # median dataset spacing
                    T.Spacingd(
                        keys=["image", "mask"],
                        pixdim=(3.32, 0.625, 0.625),
                        mode=("bilinear", "nearest"),
                    ),
                    T.ScaleIntensityRangePercentilesd(
                        keys="image", lower=0, upper=99.5, b_min=0, b_max=1
                    ),
                    # remove other labels from mask
                    CropMaskByLabel(
                        mask_key="mask",
                        label_key="ivd_label",
                        label_lambda_func=lambda x: x + 200,
                    ),
                    # some augmentations
                    T.RandGaussianNoised(
                        keys=["image"], mean=0.0, std=0.015, prob=prob
                    ),
                    T.RandRotated(
                        keys=["image", "mask"],
                        range_x=30 * (np.pi / 180),
                        mode=["bilinear", "nearest"],
                        prob=prob,
                    ),
                    # center and crop image around ivd
                    T.CropForegroundd(
                        keys=["image", "mask"],
                        source_key="mask",
                        margin=(0, 80, 80),
                        allow_smaller=False,
                    ),
                    T.CenterSpatialCropd(keys=["image", "mask"], roi_size=(-1, 80, 80)),
                    # get a single slice
                    T.CenterSpatialCropd(keys=["image", "mask"], roi_size=(5, -1, -1)),
                    T.RandSpatialCropd(keys=["image", "mask"], roi_size=(1, -1, -1)),
                    AssertEmptyImaged(),
                    # resize
                    T.Resized(
                        keys=["image", "mask"],
                        spatial_size=(1, 64, 64),
                        anti_aliasing=True,
                    ),
                    T.ToTensord(keys=["image", "mask"]),
                    T.SqueezeDimd(keys=["image", "mask"], dim=1),
                ]
            )
        else:  # Only raw images
            transforms = T.Compose(
                [
                    # each image has a single ivd label
                    T.LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
                    T.Orientationd(keys=["image", "mask"], axcodes="RAS"),
                    # median dataset spacing
                    T.Spacingd(
                        keys=["image", "mask"],
                        pixdim=(3.32, 0.625, 0.625),
                        mode=("bilinear", "nearest"),
                    ),
                    T.ScaleIntensityRangePercentilesd(
                        keys="image", lower=0, upper=99.5, b_min=0, b_max=1
                    ),
                    # remove other labels from mask
                    CropMaskByLabel(
                        mask_key="mask",
                        label_key="ivd_label",
                        label_lambda_func=lambda x: x + 200,
                    ),
                    # center and crop image around ivd
                    T.CropForegroundd(
                        keys=["image", "mask"],
                        source_key="mask",
                        margin=(0, 80, 80),
                        allow_smaller=False,
                    ),
                    T.CenterSpatialCropd(keys=["image", "mask"], roi_size=(-1, 80, 80)),
                    # get a single slice
                    T.CenterSpatialCropd(keys=["image", "mask"], roi_size=(5, -1, -1)),
                    T.RandSpatialCropd(keys=["image", "mask"], roi_size=(1, -1, -1)),
                    AssertEmptyImaged(),
                    # resize
                    T.Resized(
                        keys=["image", "mask"],
                        spatial_size=(1, 64, 64),
                        anti_aliasing=True,
                    ),
                    T.ToTensord(keys=["image", "mask"]),
                    T.SqueezeDimd(keys=["image", "mask"], dim=1),
                ]
            )
        return transforms

    def create_dataset(self):
        spider_files = self.create_spider_files()
        return data.CacheDataset(spider_files, transform=self.get_transform())


class RetinaMNISTDataset(Dataset):
    """Loads the RetinaMNIST dataset. Rescales the label to be between 0 and 1.

    Taken from https://github.com/matanat/dae_counterfactual/blob/main/config.py
    """

    LABEL_MAX = 4.0

    def create_dataset(self):
        from medmnist.dataset import RetinaMNIST

        transform_list: list = [
            ToTensor(),
        ]
        if "train" in self.mode:
            print("retinaMNIST: train augmentations")
            transform_list.extend(
                [
                    T.RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                    T.RandFlip(spatial_axis=[-1, -2], prob=0.5),
                    T.RandGridDistortion(prob=0.5),
                    T.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                    T.ScaleIntensity(),
                ]
            )

        if "ddpm" in self.mode:
            print("retinaMNIST: ddpm normalize")
            transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        transform = T.Compose(transform_list)
        return RetinaMNIST(
            split="train",
            transform=transform,
            download=True,
            as_rgb=True,
            size=128,
            root="/home/tha/datasets/medmnist",
        )

    def __init__(self, split: str = "train", mode="train", regularize_label=True):
        """Create the RetinaMNIST dataset.

        Parameters
        ----------
        split : str, optional
            Split to use for the dataset, by default "train"
        mode : str, optional
            How to configure and return datapoints, by default "train".
            If any of the following keywords are in mode, the following will be returned:
            - "train": augmented x
            - "ddpm": x with normalization to range [-1, 1] for DDPM training
            - "ace": x with empty condition for DDPM training
            - "cf": x with target (0 or 4 depending on the initial value) for counterfactual generation
        regularize_label : bool, optional
            Whether to regularize the label with a normal distribution, by default True
        """
        self.split = split
        self.mode = mode
        self.regularize_label = regularize_label
        self.reg_std = 0.05

        self.dataset = self.create_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x: torch.Tensor
        x, y = self.dataset[idx]  # (x, y), y is numpy array
        y_normed: torch.Tensor = self.transform_label(y)
        if "ace" in self.mode:
            # Return empty dict as condition
            return x, {}
        elif "cf" in self.mode:
            # Return target for CF. if y < 2 return 0, else return 4
            target = torch.Tensor([0.0 if y_normed > 0.5 else 1.0])
            return x, y_normed, target
        else:
            return x, y_normed

    def transform_label(self, y: int):
        y_pt = torch.from_numpy(y).float() / self.LABEL_MAX

        # Only regularize label during training
        if self.mode == "train" and self.regularize_label:
            noise = torch.normal(0, self.reg_std, size=y_pt.shape)
            # Use half normal distribution for value limits
            noise = torch.where(y_pt == 0.0, torch.abs(noise), noise)
            noise = torch.where(y_pt == 1.0, -torch.abs(noise), noise)
            y_pt = y_pt + noise

        return y_pt


class RetinaMNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = RetinaMNISTDataset(split="train", mode="train")
        self.val_dataset = RetinaMNISTDataset(split="val", mode="val")
        self.test_dataset = RetinaMNISTDataset(split="test", mode="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


if __name__ == "__main__":
    # apptainer binding
    # -B /home/space/datasets-sqfs/SPIDER.sqfs:/data/SPIDER:image-src=/
    # dataset = SPIDERDataset(root="/data/SPIDER", split="training")
    # print(len(dataset))
    # print(dataset[0])

    dataset = RetinaMNISTDataset()
    print(len(dataset))
    x, y = dataset[0]
    print(x, y)
    print(x.min(), x.max())
