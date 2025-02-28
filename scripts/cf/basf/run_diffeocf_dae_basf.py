import os
from lightning import seed_everything
from tqdm import tqdm
from thesis_utils.diffeocf import (
    DiffeoCF,
)

from thesis_utils.counterfactuals import (
    CFResult,
    save_cf_results,
    update_results_oracle,
)
import torch
import argparse

from thesis_utils.file_utils import (
    assert_paths_exist,
    create_result_dir,
    deterministic_run,
    dump_args,
)
from thesis_utils.generators import DAE
from thesis_utils.image_folder_dataset import default_transforms
from thesis_utils.basf_dataset import BASFDataset
from thesis_utils.models import load_resnet


def init_args():
    parser = argparse.ArgumentParser(description="Run adversarial attack.")
    parser.add_argument(
        "--gmodel_path",
        type=str,
        required=True,
        help="Path to the generative model checkpoint.",
    )
    parser.add_argument(
        "--rmodel_path",
        type=str,
        required=True,
        help="Path to the regression model.",
    )
    parser.add_argument(
        "--roracle_path",
        type=str,
        required=True,
        help="Path to the oracle model.",
    )

    def parse_target(value):
        if value == "-inf":
            return float("-inf")
        elif value == "inf":
            return float("inf")
        else:
            return float(value)

    parser.add_argument(
        "--target",
        type=parse_target,
        required=True,
        help="Target class for the attack. Can be a specific value or -inf or inf for untargeted attacks.",
    )
    parser.add_argument(
        "--stop_at",
        type=float,
        required=True,
        help="Target goal for the attack. Will stop the attack if the value is reached.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=False,
        help="Confidence to goal to stop the attack",
        default=0.05,
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
        "--num_steps",
        type=int,
        required=False,
        default=100,
        help="Number of steps for the attack.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        help="Learning rate for the optimizer.",
        default=5e-2,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help=(
            "Path to the samples folder. "
            "The folder should contain images and an optional data.csv file with two "
            "fields for filename and age."
            " If available, it will update true labels in the result dict."
        ),
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
        default="diffeocf_results_default",
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
            args.gmodel_path,
            args.rmodel_path,
            args.roracle_path,
            args.image_folder,
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generative model
    gmodel = DAE(
        "basf",
        checkpoint_path=args.gmodel_path,
        forward_t=args.forward_t,
        backward_t=args.backward_t,
    ).to(device)

    # Load regression model
    regressor = load_resnet(args.rmodel_path)

    diffeo_cf = DiffeoCF(
        gmodel=gmodel,
        rmodel=regressor,
        data_shape=(3, args.size, args.size),
        result_dir=args.result_dir,
    )

    # Save args to a config txt file
    create_result_dir(args.result_dir)
    dump_args(args, args.result_dir)

    compose = default_transforms(args.size, ddpm=True)
    dataset = BASFDataset(
        root_dir=args.image_folder,
        dataset_name="combined",
        transform=compose,
        get_mode="cf",
    )

    num_samples = (
        len(dataset)
        if args.limit_samples is None
        else min(args.limit_samples, len(dataset))
    )

    deterministic_run(0)

    diffeocf_results: list[CFResult] = []
    with tqdm(range(num_samples), desc="Running DiffeoCF") as pbar:
        for i in pbar:
            f, x, y = dataset[i]
            x = x.unsqueeze(0)

            f_basename = os.path.basename(f)

            pbar.set_postfix_str(f"Processing: {f}")

            diffeocf_result = diffeo_cf.adv_attack_dae(
                x=x,
                num_steps=args.num_steps,
                lr=args.lr,
                target=args.target,
                stop_at=args.stop_at,
                confidence_threshold=args.confidence_threshold,
                image_path=f,
            )

            diffeocf_result.update_y_true_initial(y.item())
            diffeocf_results.append(diffeocf_result)

    # Save the results for the diffeo_cf attacks
    del regressor
    del gmodel
    oracle = load_resnet(args.roracle_path).to(device)
    update_results_oracle(oracle, diffeocf_results, args.confidence_threshold)

    save_cf_results(diffeocf_results, args.result_dir)
