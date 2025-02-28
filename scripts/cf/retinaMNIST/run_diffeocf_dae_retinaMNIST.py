import os
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
from thesis_utils.healthcare_datasets import RetinaMNISTDataset
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
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=False,
        help="Confidence to goal to stop the attack",
        default=0.05,
    )

    def parse_dist(s: str):
        if s is None or s == "none":
            return None, 0
        dist, val = s.split("=")
        assert dist in [
            "l1",
            "l2",
        ], f"Invalid distance function {dist}. Only l1 or l2 are supported."
        return dist, float(val)

    parser.add_argument(
        "--dist",
        type=parse_dist,
        help="Distance function to use for the attack.",
    )
    parser.add_argument(
        "--dist_type",
        type=str,
        help="Which distance to use for the attack.",
        default="latent",
        choices=["latent", "pixel"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to use for the attack.",
        choices=["adam", "sgd", "sgd_momentum"],
        default="adam",
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
        "--limit_samples", type=int, help="Limit for the dataset.", default=None
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for the attack", default=4
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        default="dcf_dae_retinaMNIST",
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
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generative model
    gmodel = DAE(
        "retinaMNIST",
        checkpoint_path=args.gmodel_path,
        forward_t=args.forward_t,
        backward_t=args.backward_t,
    ).to(device)

    # Load regression model
    regressor = load_resnet(args.rmodel_path)

    size = 128
    dist, dist_factor = args.dist
    diffeo_cf = DiffeoCF(
        gmodel=gmodel,
        rmodel=regressor,
        data_shape=(3, size, size),
        result_dir=args.result_dir,
        dist=dist,
        dist_factor=dist_factor,
        optimizer=args.optimizer,
        dist_type=args.dist_type,
    )

    # Save args to a config txt file
    create_result_dir(args.result_dir)
    dump_args(args, args.result_dir)

    dataset = RetinaMNISTDataset(split="val", mode="cf_ddpm")

    num_samples = (
        len(dataset)
        if args.limit_samples is None
        else min(args.limit_samples, len(dataset))
    )
    dataset = torch.utils.data.Subset(dataset, range(num_samples))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    deterministic_run(0)

    diffeocf_results: list[CFResult] = []
    with tqdm(dataloader, desc="DiffeoCF RetinaMNIST") as pbar:
        for b_i, data in enumerate(pbar):
            xs, y_true, targets = data

            b_i = b_i * len(xs)  # batch index
            f_basenames = [str((b_i) + i) for i in range(len(xs))]

            diffeocf_result = diffeo_cf.adv_attack_dae_batch(
                xs=xs,
                targets=targets[: len(xs)],
                stop_ats=targets[: len(xs)],
                image_paths=f_basenames,
                confidence_threshold=args.confidence_threshold,
                num_steps=args.num_steps,
                lr=args.lr,
            )

            for j in range(len(diffeocf_result)):
                diffeocf_result[j].update_y_true_initial(y_true[j].item())

            diffeocf_results.extend(diffeocf_result)

    # Save the results for the diffeo_cf attacks
    del regressor
    del gmodel
    oracle = load_resnet(args.roracle_path).to(device)
    update_results_oracle(oracle, diffeocf_results, args.confidence_threshold)

    save_cf_results(diffeocf_results, args.result_dir)
