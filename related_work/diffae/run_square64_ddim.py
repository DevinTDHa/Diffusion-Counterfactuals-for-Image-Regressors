import sys

sys.path.append("/home/tha/master-thesis-xai/thesis_utils")
from templates import *
from templates_latent import *
import argparse

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Run square64 training")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        help="Name of the checkpoint file",
        default="last.ckpt",
    )

    # train the autoenc moodel
    # this can be run on 2080Ti's.
    conf = square64_autoenc()
    train(conf)
