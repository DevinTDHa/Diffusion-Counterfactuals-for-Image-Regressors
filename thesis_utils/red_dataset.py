import os
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torchvision import transforms
from tqdm import tqdm


class RedDataset(Dataset):
    def __init__(self, image_folder, img_size=64):
        self.image_folder = image_folder
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

        self.image_files = []
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.endswith(("png", "jpg", "jpeg")):
                    self.image_files.append(os.path.join(root, file))

        self.dirname = os.path.basename(os.path.dirname(self.image_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        red_channel = image[0, :, :]  # Red channel
        avg_red_value = torch.mean(red_channel).view(1)
        return image, avg_red_value

    def plot_distribution(self, output_folder):
        print("Plotting distribution of the first 10k images...")
        red_values = []
        max_len = min(10_000, len(self))
        for idx in tqdm(range(max_len)):
            _, avg_red_value = self[idx]
            red_values.append(avg_red_value.item())

        plt.hist(red_values, bins=255)
        plt.title(f"Red Channel Distribution for {self.dirname}")
        plt.xlabel("Average Red Value")
        plt.ylabel("Frequency")
        plt.legend(["Red Channel"])
        plt.savefig(
            os.path.join(output_folder, f"red_distribution_{self.dirname}"), dpi=200
        )


class RedDataModule(L.LightningDataModule):
    def __init__(
        self, image_folder, img_size=64, batch_size=32, num_workers=4, trainer=None
    ):
        super().__init__()
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.trainer = trainer

    def setup(self, stage=None):
        dataset = RedDataset(self.image_folder, img_size=self.img_size)
        if self.trainer:
            dataset.plot_distribution(self.trainer.log_dir)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
