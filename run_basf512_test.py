import sys

sys.path.append("/home/tha/master-thesis-xai/thesis_utils")
from templates import *
from templates_latent import *
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basf512 training")
    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    conf = basf512_autoenc_test()
    args = parser.parse_args()

    train(conf, max_time="00:01:00:00")
