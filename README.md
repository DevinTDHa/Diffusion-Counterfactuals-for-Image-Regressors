# Diffusion Counterfactuals for Image Regression

This is the official repository for the reproduction of the paper Diffusion Counterfactuals for Image Regression (Trung Duc Ha, Sidney Bender).

We provide pre-trained models that we used in the paper in Section [Pre-trained models](#pre-trained-models). For a complete reproduction of our results, see Section [Complete Reproduction](#complete-reproduction).


## Pre-trained Models

We used the following models:

- **Adversarial Counterfactual Explanations (ACE)**: Download "CelebaA HQ Diffusion Model" from the [ACE repo](https://github.com/guillaumejs2403/ACE?tab=readme-ov-file#downloading-pre-trained-models)
- **Diffusion Autoencoder (Diff-AE)**: Download "DiffAE (autoencoding only): FFHQ256" from the [Diff-AE repo](https://github.com/phizaz/diffae?tab=readme-ov-file#checkpoints)
- Following ACE, various **STEEX** models from the [STEEX repo](https://github.com/valeoai/STEEX/releases): 
    - For the classifier, download `checkpoints_decision_densenet.tar.gz`
    - For the MNAC classifier, download `checkpoints_oracle_attribute.tar.gz `
- For the FVA classifier, download `resnet50_ft` from [VGGFace2](https://github.com/cydonia999/VGGFace2-pytorch)
- Our Models (Skip these if reproducing from scratch)
    - Square dataset regression model: TODO
    - imdb-wiki-clean regression model
        - Regressor: TODO
        - Oracle: TODO

## Complete Reproduction

For a complete reproduction, we provide Python code and shell script. 

Requirements:

- We developed this code to run on an NVIDIA A100 80GB GPU. Adjust batch sizes to your available RAM.
- The code assumes that the conda environment defined in [`environment.yaml`](/environment.yaml) is active.
- The [Pre-trained models](#pre-trained-models) (except ours) need to be extracted and in the [`pretrained_models`](/pretrained_models) folder.

The reproduction code is structured as follows (assumes running from this directory)

1. `1_reproduce_all.sh`: Runs all the other reproduction scripts below.
    - By default, will use the current directory to store all the files and outputs. You can change the destinations by changing the `DCFIR_OUTPATH` variables in the script.
    - if manually running the scripts, you need to set `DCFIR_OUTPATH` manually and run [`setup_pythonpath.sh`](/setup_pythonpath.sh) to setup the necessary modules
2. `2_get_datasets.sh`: Downloads all the relevant datasets (imdb-wiki-clean, CelebA-HQ, Square).
3. `3_train_square_generators.sh`: Trains the ACE DDPM and Diff-AE for the Square dataset.
4. `4_train_regressors.sh`: Trains the regression and oracle models.
5. `5_produce_results.sh`: Runs the experiments to produce the quantitative and qualitative results.
5. `6_metrics.sh`: Calculates the metrics for the experiments.

After running the scripts, by default the results will be in `DCFIR_OUTPATH=$PWD/diff_cf_ir_results`.
