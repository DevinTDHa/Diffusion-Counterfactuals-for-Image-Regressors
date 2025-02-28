import sys
import os

sys.path.append("/home/tha/diffae/")
sys.path.append("/home/tha/master-thesis-xai/thesis_utils")
sys.path.append("/home/tha/diffeo-cf")
sys.path.append(os.getcwd())

from dae_counterfactuals.models import DAEModel


import argparse
import torch
from templates import *
from thesis_utils.models import load_model
from thesis_utils.file_utils import save_img_threaded, rename_if_exists
from counterfactuals.utils import (
    make_dir,
)

import json

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("text")

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_cfs(x_cfs, rmodel) -> dict:
    preds = rmodel(x_cfs)
    return {i: pred.item() for i, pred in enumerate(preds)}


def save_results(result_dir: str, x_cfs: torch.Tensor, results: dict) -> None:
    """
    Save the generated counterfactual images and their evaluation results.

    Parameters:
    result_dir (str): Directory to save the results.
    x_cfs (torch.Tensor): Tensor containing the generated counterfactual images.
    results (dict): Dictionary containing the evaluation results of the counterfactual images.
    """
    make_dir(result_dir)
    for i, x_cf in enumerate(x_cfs):
        save_img_threaded(
            x_cf,
            f"{result_dir}/cf_{i}.png",
        )

    with open(f"{result_dir}/results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adversarial attack.")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--resize", type=int, required=True, help="Target size of the input image."
    )
    parser.add_argument(
        "--rmodel_path",
        required=True,
        help="Path to the regression model.",
    )
    parser.add_argument(
        "--rmodel_type",
        type=str,
        required=True,
        help="Type to the regression model.",
    )
    parser.add_argument(
        "--zsem_path",
        type=str,
        required=True,
        help="Path to the saved semantic latent tensor.",
    )
    parser.add_argument(
        "--num_cfs",
        type=int,
        default=8,
        help="Number of counterfactuals to generate.",
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
        "--batch_size",
        type=int,
        default=8,
        help="Batch Size for generation",
        required=False,
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="diffeocf_diversity_results",
        help="Directory to save the results.",
        required=False,
    )

    args = parser.parse_args()
    print("Running with args:", args)

    # check path exists for all files
    for f in [args.image_path, args.rmodel_path, args.zsem_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(f, "not found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the semantic latent tensor
    z_sem = torch.load(args.zsem_path, map_location=device)

    # Load models and data info
    # Load generative model
    gmodel = DAEModel(forward_t=args.forward_t, backward_t=args.backward_t)
    x_cfs = gmodel.generate_more_cf(z_sem, args.num_cfs, args.batch_size)

    # Load regression model
    rmodel = load_model(args.rmodel_type, args.rmodel_path).to(device)
    results = evaluate_cfs(x_cfs, rmodel)

    # Save x and z in cwd for later use
    rename_if_exists(args.result_dir)
    save_results(args.result_dir, x_cfs, results)
