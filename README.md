# Diffusion Counterfactuals for Image Regression

This is the official repository for the reproduction of the paper Diffusion Counterfactuals for Image Regression (Trung Duc Ha, Sidney Bender).

We provide pre-trained models that we used in the paper in Section [Pre-trained models](#pre-trained-models). For a complete reproduction of our results, see Section [Complete Reproduction](#complete-reproduction).


## Pre-trained Models

We used the following models:

- **Adversarial Counterfactual Explanations (ACE)**: Download "CelebaA HQ Diffusion Model" from the [ACE repo](https://github.com/guillaumejs2403/ACE?tab=readme-ov-file#downloading-pre-trained-models)
- **Diffusion Autoencoder (Diff-AE)**: Download "DiffAE (autoencoding only): FFHQ256" from the [Diff-AE repo](https://github.com/phizaz/diffae?tab=readme-ov-file#checkpoints)
- Square dataset regression model: TODO
- imdb-wiki-clean regression model
    - Regressor: TODO
    - Oracle: TODO

## Complete Reproduction

For a complete reproduction, we provide Python code and shell script. 

Requirements:

1. The code assumes that the conda environment defined in [environment.yaml](/environment.yaml) is active.
2. The pre-trained ACE and Diff-AE models described in [Pre-trained models](#pre-trained-models) needs to be in the [pretrained_models](/pretrained_models) folder.

The reproduction code is structured as follows

1. `1_reproduce_all.sh`: Runs all the other reproduction scripts below.
    - By default, will use the current directory to store all the files and outputs. You can change the destinations by changing the `*_OUTPATH` variables in the script.
2. `2_get_datasets.sh`: Downloads all the relevant datasets.
3. `3_train_square_generators.sh`: Trains the ACE DDPM and Diff-AE for the square dataset.
4. `4_train_regressors.sh`: Trains the regression and oracle models.
5. `5_produce_results.sh`: Runs the experiments to produce the quantitative and qualitative results.
