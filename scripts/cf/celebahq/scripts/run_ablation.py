import os
# TODO: This is for the real reproduction file
if not os.getenv("DCFIR_OUTPATH") or not os.getenv("DCFIR_HOME"):
    print(
        "DCFIR_OUTPATH or DCFIR_HOME is not defined. Please set it manually before running this script."
    )
    exit(1)
else:
    DCFIR_OUTPATH: str = os.environ["DCFIR_OUTPATH"]
    DCFIR_HOME: str = os.environ["DCFIR_HOME"]

GMODEL_PATH = os.path.join(DCFIR_HOME, "pretrained_models/diffae/last.ckpt")
RMODEL_PATH = os.path.join(
    DCFIR_OUTPATH, "regressors/imdb-clean/version_0/checkpoints/last.ckpt"
)
RORACLE_PATH = os.path.join(
    DCFIR_OUTPATH, "regressors/imdb-clean_oracle/version_0/checkpoints/last.ckpt"
)
OUTPUT_PATH = os.path.join(DCFIR_OUTPATH, "diffae-re/ablation")
CELEBAHQ_FOLDER = os.path.join(DCFIR_OUTPATH, "datasets/CelebAMask-HQ")

# Check that the paths are correct
for path in [GMODEL_PATH, RMODEL_PATH, RORACLE_PATH, OUTPUT_PATH, CELEBAHQ_FOLDER]:
    if not os.path.exists(path):
        print(f"Path {path} does not exist. Please make sure to run the other experiments first.")
        exit(1)

def get_name(args: dict):
    return (
        f"dist={args['dist']}+{args['dist_type']}"
        f"-opt={args['optimizer']}-lr={args['lr']}"
    )

def construct_args(
    dist: str,
    dist_type: str,
    optimizer: str,
    lr: float,
):
    args = {
        "gmodel_path": GMODEL_PATH,
        "rmodel_path": RMODEL_PATH,
        "roracle_path": RORACLE_PATH,
        "confidence_threshold": 0.05,
        "dist": dist,
        "dist_type": dist_type,
        "optimizer": optimizer,
        "forward_t": 250,
        "backward_t": 10,
        "num_steps": 200,
        "lr": lr,
        "image_folder": CELEBAHQ_FOLDER,
        "limit_samples": 64,
        "batch_size": 4,
    }

    name = get_name(args)
    args["result_dir"] = os.path.join(OUTPUT_PATH, name)

    print("Running with args:", args)
    return args


def get_full_cmd(args: dict):
    args_string = " ".join([f"--{key}={value}" for key, value in args.items()])
    cmd = " ".join(
        [
            f"python {DCFIR_HOME}/scripts/cf/celebahq/run_diffeocf_dae_celebahq.py",
            args_string,
        ]
    )

    print("Running", cmd)
    return cmd


if __name__ == "__main__":
    def get_hyperparams():
        default_lr = 0.002
        default_optimizer = "adam"
        default_dist = "none"
        default_dist_type = "latent"

        dists_combinations = [
            {
                "dist": "none",
                "dist_type": "latent",
                "optimizer": default_optimizer,
                "lr": default_lr,
            }
        ]
        for dist_type in ["l1", "l2"]:
            for c in [
                "1e-6",
                "1e-5",
                "1e-4",
                "1e-3",
                "1e-2",
            ]:
                for dt in ["latent", "pixel"]:
                    dists_combinations.append(
                        {
                            "dist": f"{dist_type}={c}",
                            "dist_type": dt,
                            "optimizer": default_optimizer,
                            "lr": default_lr,
                        }
                    )

        optim_combinations = []
        for opt in ["adam", "sgd", "sgd_momentum"]:
            for lr in [0.001, 0.002, 0.005, 0.01, 0.02]:
                optim_combinations.append(
                    {
                        "dist": default_dist,
                        "dist_type": default_dist_type,
                        "optimizer": opt,
                        "lr": lr,
                    }
                )

        return dists_combinations + optim_combinations

    hyperparam_combos = get_hyperparams()
    print("Total hyperparam combos:", len(hyperparam_combos))
    for hyperparams in get_hyperparams():
        args = construct_args(**hyperparams)
        cmd = get_full_cmd(args)
        os.system(cmd)
