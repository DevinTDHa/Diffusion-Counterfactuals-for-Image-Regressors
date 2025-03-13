import os
from typing import List, Tuple
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

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from diff_cf_ir.counterfactuals import CFResult
from diff_cf_ir.generate_squares import latent_to_square_image
from diff_cf_ir.models import load_model
import pandas as pd


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
        elif self.get_mode == "ace":
            return img, {}
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


def get_experiment_targets(y_true: torch.Tensor):
    # Mirror experiment: if the values are in the lower half, we should add 0.5
    # (to get them to the other side). Otherwise, we should subtract 0.5.
    y_rgb = (y_true * 255).to(torch.uint8)
    mirrored = torch.where(y_rgb <= 127, y_rgb + 128, y_rgb - 128).to(y_true)
    return mirrored / 255


def plot_latents_with_arrows_regr(
    original_latents,
    counterfactual_latents,
    filename,
    decision_boundary,
):
    fig, ax = plt.subplots()

    # Convert to numpy arrays for easy manipulation
    original_latents = np.array(original_latents)
    counterfactual_latents = np.array(counterfactual_latents)
    y_initials = original_latents[:, 0]
    y_ends = counterfactual_latents[:, 0]

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
    # ax.set_xlabel("Square Color")
    # ax.set_ylabel("Background Color")

    # Add vertical dotted line for Foreground Color == 0.5
    plt.axvline(x=0.5, color="black", linestyle="--")

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

    # Reduce margins
    plt.tight_layout()

    # Show the plot
    plt.grid(True)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved counterfactual visualization to {filename}")


def create_decision_boundary_regr(
    predictor: torch.nn.Module, batch_size: int, device: str
) -> np.ndarray:
    # Create the grid for plotting
    num_steps = 64
    x = torch.linspace(0, 1, num_steps)
    y = torch.linspace(0, 1, num_steps)
    xx, yy = torch.meshgrid(x, y)
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    prediction_grids = []
    positions = [0, 26, 52]  # DHA: Possible square positions

    cache_file = "/tmp/decision_boundary.npy"
    if os.path.exists(cache_file):
        print("Loading cached decision boundary from " + cache_file)
        prediction_grid_mean = np.load(cache_file)
        return prediction_grid_mean
    else:
        # Create a dataset from the grid
        class GridDataset(torch.utils.data.Dataset):
            def __init__(self, grid, positions):
                self.grid = grid
                self.positions = positions

            def __len__(self):
                return len(self.grid) * len(self.positions) ** 2

            def __getitem__(self, idx):
                grid_idx = idx % len(self.grid)
                pos_idx = idx // len(self.grid)
                x_pos = self.positions[pos_idx // len(self.positions)]
                y_pos = self.positions[pos_idx % len(self.positions)]
                latent = self.grid[grid_idx]
                image = latent_to_square_image(
                    255 * float(latent[0]),
                    255 * float(latent[1]),
                    position_x=x_pos,
                    position_y=y_pos,
                )[0]
                return transforms.ToTensor()(image)

        dataset = GridDataset(grid, positions)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

        # Predict the grid values for decision boundary
        for batch in tqdm(dataloader, desc="Creating Decision Boundary"):
            predictions = predictor(batch.to(device)).detach()
            prediction_grids.append(predictions)

        # Calculate mean prediction for each grid point
        prediction_grid = torch.cat(prediction_grids, dim=0).detach().cpu()
        prediction_grid = prediction_grid.view(
            len(positions) ** 2, num_steps, num_steps
        )
        prediction_grid_mean = torch.mean(prediction_grid, dim=0).numpy()
        np.save(cache_file, prediction_grid_mean)

        return prediction_grid_mean


def global_counterfactual_visualization_squares(
    filename: str,
    regressor_path: str,
    diffeocf_results_list: list[CFResult],
    dataset_folder: str,
):
    def load_mask(mask_path: str):
        with open(mask_path, "rb") as f:
            mask = Image.open(f).convert("RGB")
            mask = transforms.ToTensor()(mask)
        return mask

    def group_squares(
        diffeocf_results_list: List[CFResult],
    ) -> tuple[
        List[List[torch.Tensor]], List[List[torch.Tensor]], List[List[torch.Tensor]]
    ]:
        # Group by tuples of (ColorA, ColorB)
        metadata = pd.read_csv(os.path.join(dataset_folder, "data.csv"))
        grouped = metadata.groupby(["ColorA", "ColorB"])["Name"].apply(list)

        # Convert to a dict
        color_groups = grouped.to_dict()
        results_dict = {result.image_name: result for result in diffeocf_results_list}

        xs = []
        x_cfs = []
        masks = []

        for _, file_names in color_groups.items():
            cur_xs = [results_dict[file_name].x[0] for file_name in file_names]
            cur_x_cfs = [results_dict[file_name].x_prime[0] for file_name in file_names]
            cur_masks = [
                load_mask(os.path.join(dataset_folder, "masks", file_name))
                for file_name in file_names
            ]

            xs.append(cur_xs)
            x_cfs.append(cur_x_cfs)
            masks.append(cur_masks)

        return xs, x_cfs, masks

    original_latents = []
    counterfactual_latents = []

    xs, x_cfs, masks = group_squares(diffeocf_results_list)

    for x_group, cf_group, mask_group in zip(xs, x_cfs, masks):
        inner_square_colors = [
            inner_square_color(x, mask) for x, mask in zip(x_group, mask_group)
        ]
        background_colors = [
            background_color(x, mask) for x, mask in zip(x_group, mask_group)
        ]

        cf_inner_square_colors = [
            inner_square_color(cf, mask) for cf, mask in zip(cf_group, mask_group)
        ]
        cf_background_colors = [
            background_color(cf, mask) for cf, mask in zip(cf_group, mask_group)
        ]

        original_latents.append(
            [np.mean(inner_square_colors), np.mean(background_colors)]
        )
        counterfactual_latents.append(
            [np.mean(cf_inner_square_colors), np.mean(cf_background_colors)]
        )

    regressor = load_model(regressor_path)
    decision_boundary = create_decision_boundary_regr(
        regressor, batch_size=128, device="cuda"
    )
    decision_boundary = np.transpose(decision_boundary, (1, 0))

    plot_latents_with_arrows_regr(
        original_latents,
        counterfactual_latents,
        filename,
        decision_boundary,
    )
