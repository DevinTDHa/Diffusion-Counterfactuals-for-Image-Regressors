import sys

sys.path.append("/home/tha/master-thesis-xai/thesis_utils")
from templates import *
from templates_latent import *
import argparse

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Run square64 latent training")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        help="Name of the latent checkpoint file",
        default="latent.pkl",
    )

    # train the latent DPM
    # NOTE: only need a single gpu
    conf = square64_autoenc()
    conf.eval_programs = ["infer"]
    # DHA: Assume pretrained. The model is loaded in eval mode.
    train(conf, mode="eval")

    # NOTE: a lot of gpus can speed up this process
    conf = square64_autoenc_latent()
    train(conf)
