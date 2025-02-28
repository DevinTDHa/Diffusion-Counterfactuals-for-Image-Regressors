import sys
import os

from lightning import seed_everything

sys.path.append("/home/tha/diffae/")
sys.path.append("/home/tha/master-thesis-xai/thesis_utils")
sys.path.append("/home/tha/diffeo-cf")
sys.path.append(os.getcwd())

import argparse
import torch
import matplotlib.pyplot as plt
from templates import *
from torch.nn import functional as F
from thesis_utils.models import load_resnet
from thesis_utils.file_utils import save_img_threaded
from counterfactuals.utils import (
    make_dir,
    torch_to_image,
    expl_to_image,
)
from counterfactuals.plot import plot_grid_part
from dae_counterfactuals.models import DAEModel, RedModel

import json

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("text")

device = "cuda" if torch.cuda.is_available() else "cpu"


class DiffeoCF:
    def __init__(
        self,
        gmodel: DAEModel,
        rmodel: torch.nn.Module,
        data_shape: tuple[int, int, int],
        result_dir: str,
        do_normalize=True,
    ):
        self.gmodel = gmodel
        self.rmodel = rmodel
        self.data_shape = data_shape
        transformations = [
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Resize((data_shape[1], data_shape[2])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
        ]
        if do_normalize:  # DDPMs
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )
        self.transforms = transforms.Compose(transformations)

        if os.path.exists(result_dir):
            timestamp = int(os.path.getmtime(result_dir))
            os.rename(result_dir, f"{result_dir}_old_{timestamp}")

        self.result_dir = make_dir(result_dir)
        self.steps_dir = make_dir(os.path.join(result_dir, "steps"))

    def adv_attack(
        self,
        attack_style: str,
        num_steps: int,
        lr: float,
        save_at: float,
        target: float,
        image_path: str,
        maximize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        prepare adversarial attack in X or Z
        run attack
        save resulting adversarial example/counterfactual
        """
        # load image
        x: torch.Tensor = self.transforms(Image.open(image_path)).to(device)

        # define parameters that will be optimized
        params = []
        if attack_style == "z":
            # define z as params for derivative wrt to z
            z_sem, xT = self.gmodel.encode(x)
            z_sem.detach()
            x_org = (x.detach().cpu().clone() + 1) / 2
            z_org = z_sem.clone()

            z_sem.requires_grad = True
            params.append(z_sem)
        else:
            raise NotImplementedError("Attack style 'x' not implemented yet.")
            # define x as params for derivative wrt x
            # x_org = x.clone()
            # x.requires_grad = True
            # params.append(x)
            # z_sem = None

        print(
            "\nRunning counterfactual search in Z ..."
            if attack_style == "z"
            else "Running conventional adv attack in X ..."
        )
        optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=0.0)

        # run the adversarial attack
        x_prime, success = self._run_adv_attack(
            x=x,
            xT=xT,
            z_sem=z_sem,
            optimizer=optimizer,
            target=target,
            attack_style=attack_style,
            save_at=save_at,
            num_steps=num_steps,
            maximize=maximize,
        )

        if not success:
            print(
                "Warning: Maximum number of iterations exceeded! Attack did not reach target value, returned None."
            )

        # save results
        image_name = image_path.split("/")[-1].split(".")[0]
        cmap_img = "jet" if self.data_shape[0] == 3 else "gray"

        # calculate heatmap as difference dx between original and adversarial/counterfactual
        # TODO: dx to original or projection?
        self.create_heatmap(
            attack_style,
            save_at,
            x_org,
            xT,
            z_org,
            x_prime,
            image_name,
            cmap_img,
        )

        return x_prime, z_sem, xT

    def create_heatmap(
        self,
        attack_style,
        save_at,
        x_org,
        xT,
        z_org,
        x_prime,
        image_name,
        cmap_img,
    ):
        heatmap = torch.abs(x_org.to(x_prime) - x_prime).sum(dim=0).sum(dim=0)

        all_images = [torch_to_image(x_org)]
        titles = ["x", "x'", "delta x"]
        cmaps = [cmap_img, cmap_img, "coolwarm"]

        if attack_style == "z":
            all_images.append(torch_to_image(self.gmodel.decode(xT, z_org)))
            titles = ["x", "g(g^{-1}(x))", "x'", "delta x"]

            cmaps = [cmap_img, cmap_img, cmap_img, "coolwarm"]

        all_images.append(torch_to_image(x_prime))
        all_images.append(expl_to_image(heatmap))

        _ = plot_grid_part(all_images, titles=titles, images_per_row=4, cmap=cmaps)
        plt.subplots_adjust(
            wspace=0.03, hspace=0.01, left=0.03, right=0.97, bottom=0.01, top=0.95
        )

        gmodel_name = (
            f"_{type(self.gmodel).__name__}" if self.gmodel is not None else ""
        )
        plt.savefig(
            self.result_dir
            + f"cf_{image_name}_{attack_style}_{gmodel_name}_save_at_{save_at}.png",
            dpi=200,
        )

    def save_intermediate_img(self, x, img_idx, n_iter, y_pred):
        """
        Saves a single image intermediate images for the attack
        """
        img_idx_path = os.path.join(self.steps_dir, f"img_{img_idx}")
        os.makedirs(img_idx_path, exist_ok=True)

        y_pred_formatted = f"{y_pred.item():.2f}"
        out_path = os.path.join(
            img_idx_path, f"niter_{n_iter:04d}_y_{y_pred_formatted}.png"
        )
        save_img_threaded(x, out_path)

    def _run_adv_attack(
        self,
        x: torch.Tensor,
        xT: torch.Tensor,
        z_sem: torch.Tensor,
        optimizer: Optimizer,
        target: float,
        attack_style: str,
        save_at: float,
        num_steps: int,
        maximize: bool,
    ) -> tuple[torch.Tensor, bool]:
        """
        run optimization process on x or z for num_steps iterations
        early stopping when save_at is reached
        """
        target_pt = torch.Tensor([[target]]).to(x.device)

        loss_fn = nn.MSELoss()

        with tqdm(total=num_steps) as progress_bar:
            for step in range(num_steps):
                optimizer.zero_grad()

                if attack_style == "z":
                    # TODO Enable batches of x?
                    x = self.gmodel.decode(xT, z_sem)

                # assert that x is a valid image
                # TODO: check the image quality. why is it so bright? we need normalize it first? Is it perhaps normalized twice?
                x.data = torch.clip(x.data, min=0.0, max=1.0)

                regression = self.rmodel(x)
                loss: torch.Tensor = loss_fn(regression, target_pt)

                self.save_intermediate_img(x, 0, n_iter=step, y_pred=regression)

                if (maximize and regression.item() > save_at) or (
                    not maximize and regression.item() < save_at
                ):
                    return x, True

                loss.backward()

                grad_magnitude = torch.norm(z_sem.grad)

                progress_bar.set_postfix(
                    regression=regression.item(),
                    loss=loss.item(),
                    grad_norm=grad_magnitude.item(),
                    gpu_alloc_GB=torch.cuda.memory_allocated() / 1e9,
                )
                progress_bar.update()

                optimizer.step()

        return x, False
        # Assume this is a JointClassifierDDPM to get the non-noised classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adversarial attack.")
    parser.add_argument(
        "--rmodel_path",
        type=str,
        required=True,
        help="Path to the regression model.",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--resize", type=int, required=True, help="Target size of the input image."
    )
    parser.add_argument(
        "--attack_style",
        type=str,
        choices=["x", "z"],
        default="z",
        help="Attack style: 'x' or 'z'.",
        required=False,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5000,
        help="Number of steps for the attack.",
        required=False,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for the optimizer.",
        default=5e-3,
        required=False,
    )
    parser.add_argument(
        "--save_at",
        type=float,
        help="Target value for early stopping threshold.",
        default=0.9,
        required=False,
    )
    parser.add_argument(
        "--target",
        type=float,
        help="Target class for the attack.",
        default=1.0,
        required=False,
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="diffeocf_results",
        help="Directory to save the results.",
        required=False,
    )
    parser.add_argument(
        "--maximize",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        help="Whether we need to maximize towards the target value",
        default=True,
        required=False,
    )
    parser.add_argument(
        "--forward_t",
        type=int,
        default=250,
        help="Number of steps for forward diffusion.",
        required=False,
    )
    parser.add_argument(
        "--backward_t",
        type=int,
        default=20,
        help="Number of steps for backwards diffusion.",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducibility.",
    )

    args = parser.parse_args()
    print("Running with args:", args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and data info
    # Load generative model
    gmodel = DAEModel(forward_t=args.forward_t, backward_t=args.backward_t)

    # Load regression model
    regressor = load_resnet(args.rmodel_path)

    diffeo_cf = DiffeoCF(
        gmodel=gmodel,
        rmodel=regressor,
        data_shape=(3, args.resize, args.resize),
        result_dir=args.result_dir,
    )

    # Set a random seed based on the current time or any other varying factor
    print("Reseeding to:", args.seed)
    seed_everything(args.seed)

    x_cf, z, xT = diffeo_cf.adv_attack(
        attack_style=args.attack_style,
        num_steps=args.num_steps,
        lr=args.lr,
        save_at=args.save_at,
        target=args.target,
        image_path=args.image_path,
        maximize=args.maximize,
    )

    # Save args to a config txt file
    with open(os.path.join(args.result_dir, "diffeocf_last_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save x and z in cwd for later use
    filename = os.path.basename(args.image_path).split(".")[0]
    save_img_threaded(x_cf, os.path.join(args.result_dir, f"{filename}_x_cf.png"))
    torch.save(x_cf, os.path.join(args.result_dir, f"{filename}_x_cf.pt"))
    torch.save(z, os.path.join(args.result_dir, f"{filename}_z.pt"))
    torch.save(xT, os.path.join(args.result_dir, f"{filename}_xT_encoded.pt"))
