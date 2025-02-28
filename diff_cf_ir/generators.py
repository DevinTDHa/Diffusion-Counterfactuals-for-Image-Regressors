from tqdm import tqdm
import yaml
import torch
from torch import nn
from experiment import LitModel
from templates import (
    basf512_autoenc,
    square64_autoenc,
    ffhq256_autoenc,
    retina128_autoenc_base,
)


class VAE(nn.Module):
    def __init__(self, config_path, model_path):
        super().__init__()
        # from models import vae_models  # TODO: Handle This later

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        vae_model_name = config["model_params"]["name"]
        model = vae_models[vae_model_name](**config["model_params"])
        print(f"Loading VAE model {vae_model_name} from", model_path)
        chkpt = torch.load(model_path)
        model.load_state_dict(
            {k.replace("model.", ""): v for k, v in chkpt["state_dict"].items()}
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device).eval()

    def forward(self, x):
        self.model.forward(x)

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        if type(z) == list:
            z = torch.vstack(z)
        return self.model.decode(z)


# class DAE(nn.Module):
#     def __init__(
#         self,
#         dataset: str,
#         checkpoint_path: str,
#         ema_model: bool = True,
#         forward_t: int = 250,
#         backward_t: int = 20,
#     ):
#         super().__init__()

#         self.forward_t = forward_t
#         self.backward_t = backward_t

#         if dataset == "basf":
#             self.conf = basf512_autoenc()
#         elif dataset == "square":
#             self.conf = square64_autoenc()
#         else:  # Assume ffhq
#             self.conf = ffhq256_autoenc()

#         assert (
#             self.conf.model_type.has_autoenc()
#         ), "Model must have a latent autoencoder."

#         model = LitModel.load_from_checkpoint(
#             checkpoint_path=checkpoint_path, conf=self.conf, map_location="cpu"
#         )

#         self.model = model.model if not ema_model else model.ema_model
#         self.model = self.model.eval()

#     def forward(self, x):
#         return self.encode(x)

#     def encode_latent(self, x):
#         z_sem: torch.Tensor = self.model.encoder.forward(x)
#         return z_sem

#     def encode_stochastic(self, x: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
#         sampler = self.conf._make_diffusion_conf(self.forward_t).make_sampler()
#         out = sampler.ddim_reverse_sample_loop(
#             self.model, x, model_kwargs={"cond": z_sem}
#         )
#         return out["sample"]

#     @torch.no_grad()
#     def encode(self, x: torch.Tensor):
#         z_sem: torch.Tensor = self.encode_latent(x)
#         xT: torch.Tensor = self.encode_stochastic(x, z_sem)
#         return z_sem, xT

#     def to(self, device):
#         self.model.to(device)
#         return self

#     def decode(self, z_sem, xT) -> torch.Tensor:
#         """
#         Renders an image based on the provided noise and optional conditioning.

#         Mostly copied from diffusion AE (render), but changed uses the right target model.

#         DAE (conditioned DDIM) backwards pass.

#         Parameters
#         ----------
#         xT : torch.Tensor
#             The input noise tensor for the rendering process.
#         z_sem : torch.Tensor, optional
#             The conditioning tensor. If provided, the rendering will be conditioned on this tensor.

#         Returns
#         -------
#         torch.Tensor
#             The rendered image tensor, with values scaled to the range [0, 1].
#         """
#         sampler = self.conf._make_diffusion_conf(self.backward_t).make_sampler(
#             grads=True
#         )

#         pred_img = sampler.sample(
#             model=self.model, noise=xT, model_kwargs={"cond": z_sem}
#         )
#         pred_img = (pred_img + 1) / 2
#         return pred_img

#     def generate_more_cf(self, z_sem: torch.Tensor, num_cfs: int) -> torch.Tensor:
#         """Generates more counterfactuals by choosing random stochastic codes (xT) as the base noise.

#         Parameters
#         ----------
#         z_sem : torch.Tensor
#             The latent tensor to condition the generation on.
#         num_cfs : int
#             The number of counterfactuals to generate

#         Returns
#         -------
#         torch.Tensor
#             The batch of counterfactuals
#         """
#         with torch.no_grad():
#             noises = torch.randn(num_cfs, 3, self.conf.img_size, self.conf.img_size).to(
#                 z_sem
#             )

#             preds = []
#             # TODO Batch size I guess
#             for i in tqdm(range(num_cfs), desc="Generating CFs"):
#                 xT = noises[i][None]
#                 pred = self.decode(xT, z_sem)
#                 preds.append(pred.detach())

#             return torch.cat(preds, dim=0)


# Use their implementation
class DAE(nn.Module):
    def __init__(
        self,
        dataset: str,
        checkpoint_path: str,
        forward_t: int = 250,
        backward_t: int = 20,
    ):
        super().__init__()

        self.forward_t = forward_t
        self.backward_t = backward_t

        if dataset == "basf":
            self.conf = basf512_autoenc()
        elif dataset == "square":
            self.conf = square64_autoenc()
        elif dataset == "retinaMNIST":
            self.conf = retina128_autoenc_base()
        else:  # Assume ffhq
            self.conf = ffhq256_autoenc()

        assert (
            self.conf.model_type.has_autoenc()
        ), "Model must have a latent autoencoder."

        model = LitModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, conf=self.conf, map_location="cpu"
        )

        self.model = model.eval()
        del self.model.model  # TODO: Make sure to use EMA model?

    def forward(self, x):
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        z_sem: torch.Tensor = self.model.encode(x)
        xT: torch.Tensor = self.model.encode_stochastic(x, z_sem)
        return z_sem, xT

    def to(self, device):
        self.model.ema_model.to(device)
        return self

    def decode(self, z_sem, xT) -> torch.Tensor:
        return self.model.render(xT, z_sem, T=self.backward_t, grads=True)

    @torch.no_grad()
    def generate_more_cf(self, z_sem: torch.Tensor, num_cfs: int) -> torch.Tensor:
        """Generates more counterfactuals by choosing random stochastic codes (xT) as the base noise.

        Parameters
        ----------
        z_sem : torch.Tensor
            The latent tensor to condition the generation on.
        num_cfs : int
            The number of counterfactuals to generate

        Returns
        -------
        torch.Tensor
            The batch of counterfactuals
        """
        noises = torch.randn(num_cfs, 3, self.conf.img_size, self.conf.img_size).to(
            z_sem
        )

        preds = []
        # TODO Batch size I guess
        for i in tqdm(range(num_cfs), desc="Generating CFs"):
            xT = noises[i][None]
            pred = self.decode(xT, z_sem)
            preds.append(pred.detach())

            return torch.cat(preds, dim=0)
