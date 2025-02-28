import os
from tqdm import tqdm
from thesis_utils.diffeocf import (
    DiffeoCF,
)

from thesis_utils.counterfactuals import (
    CFResult,
    save_cf_results,
    update_results_oracle,
    generate_collage_multitarget,
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
from thesis_utils.image_folder_dataset import ImageFolderDataset
from thesis_utils.models import load_model
from thesis_utils.debug import setup_usr1_signal_handler, usr1_signal_received


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
        help="Distance function to use for the attack in the format of (l1|l2)=(float) or none.",
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
        default=150,
        help="Number of steps for the attack.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        help="Learning rate for the optimizer.",
        default=0.002,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to the CelebAHQ dataset",
    )
    parser.add_argument(
        "--limit_samples", type=int, help="Limit for the dataset.", default=None
    )
    # parser.add_argument(
    #     "--batch_size", type=int, help="Batch size for the attack", default=4
    # )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        default="diffeocf_results_celebahq",
        help="Directory to save the results.",
    )

    args = parser.parse_args()
    print("Running with args:", args)
    return args


def load_meta(meta_path: str):
    meta = {}
    with open(meta_path) as f:
        for line in f.readlines():
            if "," not in line:
                continue
            filename, age = line.strip().split(",")
            meta[filename] = float(age) / 100

    return meta


def get_dataset(args):
    size = 256
    compose = default_transforms(size, ddpm=True)
    dataset = ImageFolderDataset(
        folder=args.image_folder,
        size=size,
        transform=compose,
    )

    num_samples = (
        len(dataset)
        if args.limit_samples is None
        else min(args.limit_samples, len(dataset))
    )
    dataset = torch.utils.data.Subset(dataset, range(num_samples))

    meta = load_meta(os.path.join(args.image_folder, "data.csv"))
    return dataset, meta


def get_batch_multitarget(dataset, i: int):
    targets = [0.1, 0.2, 0.4, 0.6, 0.8]

    f, x = dataset[i]

    def target_str(t: float):
        return f"{t*100}"

    fs = [
        f"{os.path.splitext(os.path.basename(f))[0]}_t={target_str(target)}.png"
        for target in targets
    ]
    xs = torch.broadcast_to(x, (len(targets), *x.shape))
    targets = torch.Tensor(targets).view(-1, 1)
    return f, fs, xs, targets


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

    setup_usr1_signal_handler()  # IF SLURM sents this signal, then we stop and save

    # Load generative model
    gmodel = DAE(
        "ffhq",
        checkpoint_path=args.gmodel_path,
        forward_t=args.forward_t,
        backward_t=args.backward_t,
    ).to(device)

    # Load regression model
    regressor = load_model(args.rmodel_path)

    size = 256
    dist, dist_factor = args.dist
    diffeo_cf = DiffeoCF(
        gmodel=gmodel,
        rmodel=regressor,
        data_shape=(3, size, size),
        result_dir=args.result_dir,
        dist=dist,
        optimizer=args.optimizer,
        dist_type=args.dist_type,
        dist_factor=dist_factor,
    )

    # Save args to a config txt file
    create_result_dir(args.result_dir)
    dump_args(args, args.result_dir)

    dataset, meta = get_dataset(args)

    deterministic_run(0)

    diffeocf_results_list: list[CFResult] = []
    with tqdm(range(len(dataset)), desc="DAE Multi-target") as pbar:
        for i in pbar:
            real_name, fs, xs, targets = get_batch_multitarget(dataset, i)
            fs_basenames = [os.path.basename(f) for f in fs]
            pbar.set_postfix_str(f"Processing: {fs_basenames}")

            diffeocf_results = diffeo_cf.adv_attack_dae_batch(
                xs=xs,
                targets=targets,
                stop_ats=targets,
                image_paths=fs,
                confidence_threshold=args.confidence_threshold,
                num_steps=args.num_steps,
                lr=args.lr,
            )

            for result in diffeocf_results:
                y_true = meta[os.path.basename(real_name)]
                result.update_y_true_initial(y_true)

            diffeocf_results_list.extend(diffeocf_results)
            if usr1_signal_received():
                print("Received signal to stop. Saving results.")
                break

    # Save the results for the diffeo_cf attacks
    del regressor
    del gmodel
    oracle = load_model(args.roracle_path).to(device)
    update_results_oracle(oracle, diffeocf_results_list, args.confidence_threshold)

    save_cf_results(diffeocf_results_list, args.result_dir)

    num_targets = len(targets)
    num_batches = len(diffeocf_results_list) // num_targets
    for b_i in range(num_batches):
        cur_results = diffeocf_results_list[b_i * num_targets : (b_i + 1) * num_targets]
        generate_collage_multitarget(args.result_dir, cur_results)
