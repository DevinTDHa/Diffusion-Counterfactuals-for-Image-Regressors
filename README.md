# Diffusion Counterfactuals for Image Regressors

This is the official repository for the reproduction of the paper Diffusion Counterfactuals for Image Regressors (Trung Duc Ha, Sidney Bender).

We provide pre-trained models that we used in the paper in Section [Pre-trained models](#pre-trained-models). For a complete reproduction of our results, see Section [Complete Reproduction](#complete-reproduction).


## Pre-trained Models

We used the following models:

- **Adversarial Counterfactual Explanations (ACE)**: Download "CelebaA HQ Diffusion Model" from the [ACE repo](https://github.com/guillaumejs2403/ACE?tab=readme-ov-file#downloading-pre-trained-models)
- **Diffusion Autoencoder (Diff-AE)**: Download "DiffAE (autoencoding only): FFHQ256" from the [Diff-AE repo](https://github.com/phizaz/diffae?tab=readme-ov-file#checkpoints)
- Following ACE, various **STEEX** models from the [STEEX repo](https://github.com/valeoai/STEEX/releases): 
    - For the classifier, download `checkpoints_decision_densenet.tar.gz`
    - For the MNAC classifier, download `checkpoints_oracle_attribute.tar.gz `
- For the FVA classifier, download `resnet50_ft` from [VGGFace2](https://github.com/cydonia999/VGGFace2-pytorch)
- Our Models (Skip these if reproducing from scratch), namely the Square generators and regression models (square and imdb-wiki-clean) 
    - https://tubcloud.tu-berlin.de/s/5qMJAkXPtiW6ozg

## Complete Reproduction

For a complete reproduction, we provide Python code and shell script. 

Requirements:

- We developed this code to run on an NVIDIA A100 80GB GPU. Adjust batch sizes to your available RAM.
- The code assumes that the conda environment defined in [`environment.yaml`](/environment.yaml) is active.
- The [Pre-trained models](#pre-trained-models) (except ours) need to be extracted and in the [`pretrained_models`](/pretrained_models) folder.

The reproduction code is structured as follows (assumes running from this directory)

1. `1_reproduce_all.sh`: Runs all the other reproduction scripts below.
    - By default, will use the current directory to store all the files and outputs. You can change the destinations by changing the `DCFIR_OUTPATH` variables in the script.
    - if manually running the scripts, you need to set `DCFIR_OUTPATH` and `DCFIR_HOME` manually and  [`source setup_pythonpath.sh`](/setup_pythonpath.sh) to setup the necessary modules
2. `2_get_datasets.sh`: Downloads all the relevant datasets (imdb-wiki-clean, CelebA-HQ, Square).
3. `3_train_square_generators.sh`: Trains the ACE DDPM and Diff-AE for the Square dataset.
4. `4_train_regressors.sh`: Trains the regression and oracle models.
5. `5_produce_results.sh`: Runs the experiments to produce the quantitative and qualitative results.
5. `6_metrics.sh`: Calculates the metrics for the experiments.

After running the scripts, by default the results will be in `DCFIR_OUTPATH=$PWD/diff_cf_ir_results`.

# Citations

If you find our work useful, we would greatly appreciate it if you consider citing our and the related works:

Our Paper

```
TODO
```

A.-K. Dombrowski, J. E. Gerken, K.-R. Müller, and P. Kessel, “Diffeomorphic Counterfactuals With Generative Models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 3257–3274, May 2024, doi: 10.1109/TPAMI.2023.3339980.

```
@article{dombrowskiDiffeomorphicCounterfactualsGenerative2024,
  title = {Diffeomorphic {{Counterfactuals With Generative Models}}},
  author = {Dombrowski, Ann-Kathrin and Gerken, Jan E. and Müller, Klaus-Robert and Kessel, Pan},
  date = {2024-05},
  journaltitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume = {46},
  number = {5},
  pages = {3257--3274},
  issn = {1939-3539},
  doi = {10.1109/TPAMI.2023.3339980},
  eventtitle = {{{IEEE Transactions}} on {{Pattern Analysis}} and {{Machine Intelligence}}},
}
```

G. Jeanneret, L. Simon, and F. Jurie, “Adversarial Counterfactual Visual Explanations,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada: IEEE, Jun. 2023, pp. 16425–16435. doi: 10.1109/CVPR52729.2023.01576.
```
@inproceedings{jeanneretAdversarialCounterfactualVisual2023,
  title = {Adversarial {{Counterfactual Visual Explanations}}},
  booktitle = {2023 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  author = {Jeanneret, Guillaume and Simon, Loïc and Jurie, Frédéric},
  date = {2023-06},
  pages = {16425--16435},
  publisher = {IEEE},
  location = {Vancouver, BC, Canada},
  doi = {10.1109/CVPR52729.2023.01576},
  eventtitle = {2023 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  isbn = {979-8-3503-0129-8},
  langid = {english},
}
```

K. Preechakul, N. Chatthee, S. Wizadwongsa, and S. Suwajanakorn, “Diffusion Autoencoders: Toward a Meaningful and Decodable Representation,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA: IEEE, Jun. 2022, pp. 10609–10619. doi: 10.1109/CVPR52688.2022.01036.
```
@inproceedings{preechakulDiffusionAutoencodersMeaningful2022,
  title = {Diffusion {{Autoencoders}}: {{Toward}} a {{Meaningful}} and {{Decodable Representation}}},
  shorttitle = {Diffusion {{Autoencoders}}},
  booktitle = {2022 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  author = {Preechakul, Konpat and Chatthee, Nattanat and Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
  date = {2022-06},
  pages = {10609--10619},
  publisher = {IEEE},
  location = {New Orleans, LA, USA},
  doi = {10.1109/CVPR52688.2022.01036},
  eventtitle = {2022 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  isbn = {978-1-66546-946-3},
  langid = {english},
}
```
