import json

import torch
from tqdm import tqdm
from diff_cf_ir.diffeocf import (
    DiffeoCF,
)
from diff_cf_ir.counterfactuals import (
    CFResult,
    save_cf_results,
    update_results_true_latents,
)
from diff_cf_ir.file_utils import (
    assert_paths_exist,
    create_result_dir,
    deterministic_run,
    dump_args
)
from diff_cf_ir.generators import VAE
import argparse

from diff_cf_ir.models import load_resnet
from diff_cf_ir.squares_dataset import (
    global_counterfactual_visualization_squares,
    SquaresDataset,
    get_experiment_targets,
    inner_square_color,
)

import os


def init_args():
    parser = argparse.ArgumentParser(description="Run adversarial attack.")
    parser.add_argument(
        "--gmodel_config",
        type=str,
        required=True,
        help="Path to the config file for the generative model.",
    )
    parser.add_argument(
        "--gmodel_path",
        type=str,
        required=True,
        help="Path to the generative model.",
    )
    parser.add_argument(
        "--rmodel_path",
        type=str,
        required=True,
        help="Path to the regression model.",
    )
    parser.add_argument(
        "--attack_style",
        type=str,
        choices=["x", "z"],
        required=True,
        default="z",
        help="Attack style: 'x' or 'z'.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        required=True,
        default=500,
        help="Number of steps for the attack.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Learning rate for the optimizer.",
        default=5e-2,
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target class for the attack. Can either be mirror or a float.",
        default="mirror",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=True,
        help="Target confidence to stop the attack",
        default=0.05,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--size", type=int, required=True, help="Target size of the input image."
    )
    parser.add_argument(
        "--limit_samples", type=int, help="Limit for the dataset.", default=None
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        default="diffeocf_square_vqvae",
        help="Directory to save the results.",
    )

    args = parser.parse_args()
    print("Running with args:", args)
    return args


if __name__ == "__main__":
    args = init_args()

    # Check if all files exist
    assert_paths_exist(
        [
            args.gmodel_config,
            args.gmodel_path,
            args.rmodel_path,
            args.image_folder,
            "decision_boundary.npy",
        ]
    )
    # Load models and data info
    # Load generative model
    gmodel = VAE(args.gmodel_config, args.gmodel_path)

    # Load regression model
    print("Loading regression model from path:", args.rmodel_path)
    regressor = load_resnet(args.rmodel_path)

    # Load Square Dataset
    square_dataset = SquaresDataset(root=args.image_folder, mask_mode=True)

    diffeocf_results: list[CFResult] = []
    masks = []

    diffeo_cf = DiffeoCF(
        gmodel=gmodel,
        rmodel=regressor,
        data_shape=(3, args.size, args.size),
        result_dir=os.path.join(args.result_dir),
    )

    # Save args to a config txt file
    create_result_dir(args.result_dir)
    dump_args(args, args.result_dir)

    deterministic_run(seed=0)

    num_samples = (
        len(square_dataset)
        if args.limit_samples is None
        else min(args.limit_samples, len(square_dataset))
    )
    with tqdm(range(num_samples), desc="Running DiffeoCF") as pbar:
        for i in pbar:
            f, _ = square_dataset.data[i]
            x, y, mask = square_dataset[i]
            x = x.unsqueeze(0)

            pbar.set_postfix_str(f"Processing: {f}")

            target: float = (
                get_experiment_targets(y).item()
                if args.target == "mirror"
                else float(args.target)
            )
            diffeocf_result = diffeo_cf.adv_attack(
                x=x,
                attack_style=args.attack_style,
                num_steps=args.num_steps,
                lr=args.lr,
                target=target,
                confidence_threshold=args.confidence_threshold,
                image_path=f,
            )
            diffeocf_result.update_y_true_initial(y.item())

            diffeocf_results.append(diffeocf_result)
            masks.append(mask)

    xs = [res.x[0] for res in diffeocf_results]
    xs_reconstructed = [res.x_reconstructed[0] for res in diffeocf_results]
    x_cfs = [res.x_prime[0] for res in diffeocf_results]
    y_initials = torch.Tensor([res.y_initial_pred for res in diffeocf_results])
    y_ends = torch.Tensor([res.y_final_pred for res in diffeocf_results])

    # Update with true final predictions
    update_results_true_latents(
        [inner_square_color(x_cf, mask).item() for x_cf, mask in zip(x_cfs, masks)],
        diffeocf_results,
        args.confidence_threshold,
    )
    save_cf_results(diffeocf_results, args.result_dir)

    global_counterfactual_visualization_squares(
        filename=os.path.join(args.result_dir, "counterfactual_visualization.png"),
        input_imgs=xs,
        counterfactuals=x_cfs,
        y_initials=y_initials,
        y_ends=y_ends,
        hints=masks,
    )
