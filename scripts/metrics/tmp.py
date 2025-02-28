import os
import torch

from diff_cf_ir.counterfactuals import CFResult
from tqdm import tqdm


if __name__ == "__main__":
    res_path = "/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/celebahq/celebahqcf_dae_res=2025-01-02/imgs=CelebAMask-HQ_dist=l2_lr=0.0039_bt=15_opt=adam"

    cf_results: list[CFResult] = torch.load(os.path.join(res_path, "cf_results.pt"))

    cf_folder = os.path.join(res_path, "cf")
    os.makedirs(cf_folder, exist_ok=True)
    for cf_result in tqdm(cf_results):
        cf_result.save_cf(cf_folder)
