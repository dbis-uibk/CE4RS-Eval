# CE4RS-Eval
Counterfactual Explanation for Recommender Systems - Evaluation 

## Repository

This repository contains code of the paper "A Closer Look at Counterfactual Explanation Metrics for Recommender Systems" paper. We have evaluated our claim on three publicly available benchmarks, MovieLens1M, a subset of Yahoo!Music dataset and a subset of Pinterest dataset, using two different recommenders, Matric Factorization (MF) and Variational Auto Encoder (VAE). 

## Folders

* **Experiments Results**: contains all the results of recommenders we used for the tables and figures in the paper and other configurations discussed in paper.
* **code**: contains several code files:
  - data_processing - code related to the preprocessing step for preparing data to run with our models.
  - recommenders_architecture - specifies the architecture of the recommenders that were used in the paper(MF, VAE).
  - recommenders_training - contains code related to VAE and MLP recommenders training.
  - LXR_training - contains code for training LXR model for explaining a specified recommender(This is the only recommender that needs training).
  - metrics - contains code related to model evaluation based on baseline methods approach.
  - metricsTopK.py - code for evaluation of of explainers based on K-th item of recommender list to address consistency
  - metricsXpSize.py - code for evaluation of methods on different size values (Explanation Sparsity Metric). 
  - help_functions - includes the framework's functions that are being used in all codes.
* **checkpoints**: It is the designated location for saving and loading the trained model's checkpoints.
  
## Requirements

* python 3.10
* Pytorch 1.13
* wandb 0.16.3 (the package we used for monitoring the train process)

## Usage

To use this code, follow these steps:
+ Create data to work with by running the data_processing code.
+ On every code, please specify the "data_name" variable to be 'ML1M'/'Yahoo'/'Pinterest', and the "recommender_name" variable to be 'MLP'/'VAE' or pass it through arguments of "recommender" and "data"

## Reproducing the Results:
+ After running the preprocessing step, simply run the recommenders_training.py and specify the "data_name" variable to be 'ML1M'/'Yahoo'/'Pinterest', and the "recommender_name" variable to be 'MLP'/'VAE'.
+ From the output checkpoints check which recommenders you want to pick for explanation. Then set the file name of the checkpoint in LXR_training.py or pass it as a argument by --directory and run to train the explainers. 
+ Then to get other explainers and evaluate LXR evaluation, run the metrics.py file. This will print all the numbers you want. We have all these outputs in "Experiments Results" folder.

## Resutls

![RecLengthFig](https://github.com/dbis-uibk/CE4RS-Eval/blob/main/Resutls/Figures/TopK.png)
![XpSizeFig](https://github.com/dbis-uibk/CE4RS-Eval/blob/main/Resutls/Figures/xpSize.png)

![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/MLP%20ML-1M.png)



