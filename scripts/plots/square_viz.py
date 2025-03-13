import argparse
from PIL import Image
from torchvision.transforms import transforms
from typing import List
import os

from diff_cf_ir.counterfactuals import (
    CFResult,
)
import torch
from diff_cf_ir.squares_dataset import (
    global_counterfactual_visualization_squares,
)


def main(
    parent_folder: str, dataset_folder: str, regressor_path: str, output_file: str
):
    cf_results_path = os.path.join(parent_folder, "cf_results.pt")

    diffeocf_results_list: List[CFResult] = torch.load(
        cf_results_path, map_location="cpu"
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    global_counterfactual_visualization_squares(
        filename=output_file,
        regressor_path=regressor_path,
        diffeocf_results_list=diffeocf_results_list,
        dataset_folder=dataset_folder,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates visualizations for the square mirror experiment"
    )
    parser.add_argument(
        "result_folder",
        type=str,
        help="Parent folder of the results. It should contain the `cf_results.pt` file.",
    )
    parser.add_argument(
        "dataset_folder",
        type=str,
        help="Folder to the square dataset",
    )
    parser.add_argument(
        "regressor_path",
        type=str,
        help="Path to the regressor model checkpoint for the decision surface.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output file for the visualization. Passed to matplotlib.",
    )
    args = parser.parse_args()

    main(args.result_folder, args.dataset_folder, args.regressor_path, args.output_file)
