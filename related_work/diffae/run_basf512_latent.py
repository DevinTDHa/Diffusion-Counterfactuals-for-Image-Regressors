from templates import *
from templates_latent import *

if __name__ == "__main__":
    # do run the run_basf512 before using the file to train the latent DPM

    # infer the latents for training the latent DPM
    # TODO: Need to put this in a separate script, otherwise OOM
    # NOTE: not gpu heavy, but more gpus can be of use!
    # conf = basf512_autoenc()
    # conf.eval_programs = ["infer"]
    # # DHA: Assume pretrained. The model is loaded in eval mode.
    # train(conf, mode="eval")

    # train the latent DPM
    # NOTE: only need a single gpu
    conf = basf512_autoenc_latent()
    train(conf)
