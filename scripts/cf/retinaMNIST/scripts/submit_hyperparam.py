import os
import itertools


GMODEL_PATH = "/home/tha/python_repos/matanat_dae_counterfactual/pretrained/retina128_epoch=9259-step=1250000.ckpt"
RORACLE_PATH = "/home/tha/thesis_runs/regressor/retinaMNIST_oracle-128/version_0/checkpoints/last.ckpt"


def construct_args(rmodel, dist, lr):
    args = {
        "gmodel_path": GMODEL_PATH,
        "rmodel_path": rmodel,
        "roracle_path": RORACLE_PATH,
        "confidence_threshold": 0.05,
        "dist": dist,
        "optimizer": "adam",
        "forward_t": 250,
        "backward_t": 15,
        "num_steps": 200,
        "lr": lr,
        "limit_samples": 100,
        "batch_size": 10,
    }

    rmodel_tag = "reg" if "retinaMNIST_reg" in rmodel else "normal"
    name = f"retinaMNIST-lr={lr}-dist={dist}-rmodel={rmodel_tag}"
    args["result_dir"] = f"/home/tha/thesis_runs/dae/retinaMNIST/hyperparam/{name}"

    print("Running with args:", args)
    return args


def get_full_sbatch_cmd(args: dict):
    sbatch_script_header = f"""#!/bin/bash
#SBATCH --job-name=dae_retinaMNIST_hyper
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%j-%x.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/retinaMNIST
#SBATCH --signal=SIGUSR1@600

"""

    args_string = " ".join([f"--{key}={value}" for key, value in args.items()])
    cmd = " ".join(
        [
            "apptainer run",
            "-B /home/space/datasets:/home/space/datasets",
            "--nv",
            "~/apptainers/thesis.sif",
            "python run_diffeocf_dae_retinaMNIST.py",
            args_string,
        ]
    )

    print("Apptainer command:", cmd)
    sbatch_cmd = sbatch_script_header + cmd

    # Use a here document to pass the whole pseudo script to sbatch
    return f"sbatch <<'EOF'\n{sbatch_cmd}\nEOF"


if __name__ == "__main__":
    # Define the hyperparameters
    # lrs = [0.001, 0.002, 0.005]
    lrs = [0.002, 0.005, 0.01, 0.02]
    dists = ["l1=1e-5", "none"]
    rmodel_paths = [
        "/home/tha/thesis_runs/regressor/retinaMNIST_reg-128/version_0/checkpoints/last.ckpt",
        "/home/tha/thesis_runs/regressor/retinaMNIST_resnet18-128/version_0/checkpoints/last.ckpt",
    ]

    # Iterate through all combinations of hyperparameters
    for lr, dist, rmodel_path in itertools.product(lrs, dists, rmodel_paths):
        command = get_full_sbatch_cmd(construct_args(rmodel_path, dist, lr))
        os.system(command)
