import lightning as L
import pandas as pd
import torch
from torchvision import transforms as T
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader

from PIL import Image
import os

from tqdm import tqdm


class BASFDataset(VisionDataset):
    """Loads the BASF Dataset with either soybean or cotton images or combined images.

    The expected input folder structure is as follows:
    root
    ├── annotation_combined.txt
    ├── annotation_soybean.txt
    ├── annotation_cotton.txt
    ├── imgs
    │   ├── {UUID}.jpg
    │   ├── {DATE}.jpg
    │   ├── ...
    ├── imgs_resize512
    │   ├── {UUID}.jpg
    │   ├── {DATE}.jpg
    │   ├── ...

    """

    dataset_classes = ["combined", "soybean", "cotton"]

    # Dataset Metadata
    # image_size = [3, 512, 512]
    img_folder = "imgs_resize512"

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        normalize_label=True,
        keep_in_memory: bool = False,
        transform=T.ToTensor(),
        get_mode="regr",
    ):
        self.root: str = root_dir
        self.transform = transform
        self.normalize_label = normalize_label
        # self.label_max_value = 52  # old
        self.label_max_value = 100
        self.get_mode = get_mode

        assert (
            dataset_name in self.dataset_classes
        ), f"Invalid class label in {dataset_name}"

        # self.image_size = image_size

        annotation_file = os.path.join(self.root, f"annotation_{dataset_name}.txt")
        self.annotation_df = pd.read_csv(annotation_file, sep="\t")
        if self.normalize_label:
            self.annotation_df["Label"] = (
                self.annotation_df["Label"] / self.label_max_value
            )

        self.keep_in_memory = keep_in_memory

        if self.keep_in_memory:
            print("Loading dataset into memory")
            self.data = []
            for idx in tqdm(range(len(self.annotation_df))):
                img, label = self.load_image(idx)
                self.data.append((img, label))

    def load_image(self, index):
        base_name, label, _ = self.annotation_df.iloc[index]
        # DHA: Assume preprocessed images
        img_path = os.path.join(self.root, self.img_folder, base_name)

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        label = torch.Tensor([label])

        if self.transform is not None:
            img = self.transform(img)

        if self.get_mode == "dae":  # Diffusion Autoencoder
            return {"img": img, "index": index, "labels": label}
        if self.get_mode == "cf":  # DiffeoCF
            return base_name, img, label
        else:
            return img, label

    def __getitem__(self, index):
        if self.keep_in_memory:
            return self.data[index]
        else:
            return self.load_image(index)

    def __len__(self):
        return len(self.annotation_df)


class BASFDataModule(L.LightningDataModule):
    def __init__(
        self,
        folder_path: str,
        dataset_name: str,
        transform=[T.ToTensor()],
        batch_size=32,
        num_workers=4,
    ):
        """Loads the folder for the CelebA dataset with landmarks as features.

        root
        ├── list_landmarks_align_celeba.csv
        ├── img_align_celeba
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   ├── ...
        """
        super().__init__()

        self.folder_path = folder_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.batch_size = batch_size

        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = BASFDataset(
            root_dir=self.folder_path,
            dataset_name=self.dataset_name,
            transform=self.transform,
        )
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset,
            [0.8, 0.2],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def total_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
