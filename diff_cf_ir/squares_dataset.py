import os
from typing import Tuple
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import VisionDataset
import torch
import csv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import lightning as L


class SquaresDataset(VisionDataset):
    img_folder = "imgs"
    masks_folder = "masks"

    def __init__(
        self, label="ColorA", mask_mode=False, get_mode="regr", *args, **kwargs
    ):
        """Loads the square data folder. Expected format:

        root
        ├── data.csv
        ├── imgs
        │   ├── 0.png
        │   ├── ...
        ├── masks
        │   ├── 0.png
        │   ├── ...
        """
        super().__init__(*args, **kwargs)

        self.data: list[Tuple[str, float]] = []
        csv_path = os.path.join(self.root, "data.csv")
        with open(csv_path, mode="r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.data.append((row["Name"], float(row[label])))

        self.mask_mode = mask_mode
        self.get_mode = get_mode

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    def load_image(self, index):
        base_name, label = self.data[index]
        # DHA: Assume preprocessed images
        img_path = os.path.join(self.root, self.img_folder, base_name)

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        label = torch.Tensor([label])
        img = self.transform(img)

        if self.get_mode == "dae":  # Diffusion Autoencoder
            return {"img": img, "index": index, "labels": label}
        elif self.mask_mode:
            mask_path = os.path.join(self.root, self.masks_folder, base_name)
            with open(mask_path, "rb") as f:
                mask = Image.open(f).convert("RGB")
                mask = transforms.ToTensor()(mask)

            return base_name, img, label, mask
        else:
            return img, label

    def __getitem__(self, index):
        return self.load_image(index)

    def __len__(self):
        return len(self.data)


class SquaresDataModule(L.LightningDataModule):
    def __init__(self, folder_path, transform=None, batch_size=32):
        """Loads the square data folder. Expected format:

        root
        ├── data.csv
        ├── imgs
        │   ├── 0.png
        │   ├── ...
        ├── masks
        │   ├── 0.png
        │   ├── ...
        """
        super(SquaresDataModule, self).__init__()
        self.folder_path = folder_path
        self.csv_path = os.path.join(folder_path, "data.csv")
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        )
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = SquaresDataset(root=self.folder_path, transform=self.transform)
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def total_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=4,
        )


def inner_square_color(x: torch.Tensor, mask: torch.Tensor):
    """
    Checks the red values of the inner square, given the hint mask.
    """
    mask = mask.to(torch.bool)
    mask[[1, 2], :, :] = False  # Only select red channel

    if mask.sum().item() != 64:
        print("WARNING: Mask sum is not 64 (square should be 8x8).")

    intensity_foreground = x[mask].mean()
    return intensity_foreground


def background_color(x: torch.Tensor, mask: torch.Tensor):
    """
    Computes the background color of the square image (inverse of the square mask and for all channels).
    """
    mask = mask.to(torch.bool)
    mask = ~mask  # Invert mask

    intensity_background = x[mask].mean()
    return intensity_background


def plot_latents_with_arrows_regr(
    original_latents,
    counterfactual_latents,
    filename,
    y_initials,
    y_ends,
    decision_boundary,
):
    fig, ax = plt.subplots()

    # Convert to numpy arrays for easy manipulation
    original_latents = np.array(original_latents)
    counterfactual_latents = np.array(counterfactual_latents)
    y_initials = np.array(y_initials)
    y_ends = np.array(y_ends)

    # Regression color Maps
    pred_cmap = cm.get_cmap("inferno")  # blue to red
    decision_cmap = cm.get_cmap("inferno")

    # Display the decision boundary grid as the background
    ax.imshow(
        decision_boundary,
        extent=[0, 1, 0, 1],  # Extend from x=0 to x=1 and y=0 to y=1
        origin="lower",  # Aligns the grid with the bottom-left of the plot
        cmap=decision_cmap,  # Apply custom colormap
        alpha=0.5,  # Make the background semi-transparent
    )

    # Plot original latents and counterfactuals
    for i, (orig, cf, start_conf, end_conf) in enumerate(
        zip(
            original_latents,
            counterfactual_latents,
            y_initials,
            y_ends,
        )
    ):
        # Get the color from the colormap based on confidence (blue -> red)
        start_color = pred_cmap(start_conf)  # Color for the original point
        end_color = pred_cmap(end_conf)  # Color for the counterfactual point

        # Plot original point with darkblue border
        ax.scatter(
            orig[0],
            orig[1],
            facecolor=start_color,
            edgecolor="darkblue",
            label="Original" if i == 0 else "",
        )

        # Plot counterfactual point with darkred border
        ax.scatter(
            cf[0],
            cf[1],
            facecolor=end_color,
            edgecolor="darkred",
            label="Counterfactual" if i == 0 else "",
        )

        # Draw arrow between original and counterfactual points
        ax.annotate(
            "",
            xy=(cf[0], cf[1]),
            xytext=(orig[0], orig[1]),
            arrowprops=dict(
                fc="green", ec="green", edgecolor="yellow", arrowstyle="->", alpha=0.5
            ),
        )

    # Setting limits for the plot (0 to 1 for both axes)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Set ticks at 0.5 intervals
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))

    # Axis labels
    ax.set_xlabel("Foreground Intensity")
    ax.set_ylabel("Background Intensity")

    # Add vertical dotted line for Foreground Intensity == 0.5
    plt.axvline(x=0.5, color="black", linestyle="--")
    # plt.text(
    #     1.05,
    #     0.5,
    #     "Confounding feature only",
    #     rotation=270,
    #     verticalalignment="center",
    # )

    # Add horizontal dotted line for Background Intensity == 0.5
    plt.axhline(y=0.5, color="black", linestyle="--")
    # plt.text(0.5, 1.05, "True feature only", horizontalalignment="center")

    # Create neutral markers for the legend (gray fill color)
    handles, labels = ax.get_legend_handles_labels()
    neutral_marker = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="darkblue",
        markersize=8,
        label="Original",
    )
    cf_marker = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="darkred",
        markersize=8,
        label="Counterfactual",
    )
    by_label = dict(zip(labels, handles))

    # Override handles with neutral markers
    by_label["Original"] = neutral_marker
    by_label["Counterfactual"] = cf_marker

    # Display the updated legend
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Show the plot
    plt.grid(True)
    plt.savefig(filename, dpi=200)
    print(f"Saved counterfactual visualization to {filename}")


def global_counterfactual_visualization_squares(
    filename: str,
    input_imgs: list[torch.Tensor],
    counterfactuals: list[torch.Tensor],
    y_initials: torch.Tensor,
    y_ends: torch.Tensor,
    hints: list[torch.Tensor],
):
    original_latents = []
    counterfactual_latents = []

    for x, cf, hint in zip(input_imgs, counterfactuals, hints):
        original_latents.append(
            [inner_square_color(x, hint), background_color(x, hint)]
        )
        counterfactual_latents.append(
            [
                inner_square_color(cf, hint),
                background_color(cf, hint),
            ]
        )

    path = os.path.join("decision_boundary.npy")
    assert os.path.exists(
        path
    ), f"Decision boundary {path} file not found. Should be in current directory."

    decision_boundary = np.load(path)
    decision_boundary = np.transpose(decision_boundary, (1, 0))

    plot_latents_with_arrows_regr(
        original_latents,
        counterfactual_latents,
        filename,
        y_initials,
        y_ends,
        decision_boundary,
    )


def get_experiment_targets(y_true: torch.Tensor):
    # Mirror experiment: if the values are in the lower half, we should add 0.5
    # (to get them to the other side). Otherwise, we should subtract 0.5.
    y_rgb = (y_true * 255).to(torch.uint8)
    mirrored = torch.where(y_rgb <= 127, y_rgb + 128, y_rgb - 128).to(y_true)
    return mirrored / 255
