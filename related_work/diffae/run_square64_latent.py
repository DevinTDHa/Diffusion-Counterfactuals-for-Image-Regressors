import sys


from templates import *
from templates_latent import *
import argparse

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Run square64 latent training")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the DDIM.",
    )
    args = parser.parse_args()

    # train the latent DPM
    # NOTE: only need a single gpu
    conf = square64_autoenc()
    conf.base_dir = args.checkpoint_path
    conf.eval_programs = ["infer"]
    # DHA: Assume pretrained. The model is loaded in eval mode.
    train(conf, mode="eval")

    # NOTE: a lot of gpus can speed up this process
    conf = square64_autoenc_latent()
    conf.base_dir = args.checkpoint_path
    train(conf)
