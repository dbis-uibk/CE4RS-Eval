# CE4RS-Eval
### Counterfactual Explanation for Recommender Systems - Evaluation 
In this paper, we critically examine the evaluation of counterfactual explainers through consistency and explanation sparsity as key principles of effective explanation.
Through extensive experiments, we assess how incorporating Top-k recommendations impacts the consistency of existing evaluation metrics; and analyze the impact of explanation size on explainer's performance, highlighting its importance as a key determinant of explanation quality.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Repository

This repository contains code of the paper "Beyond Top-1: Mitigating Inconsistency in the Evaluation of Counterfactual Explanations for Recommender Systems" paper. We have evaluated our claim on three publicly available benchmarks, MovieLens1M, a subset of Yahoo!Music dataset and a subset of Pinterest dataset, using two different recommenders, Matric Factorization (MF) and Variational Auto Encoder (VAE). 

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

* ## Installation
Main libraries:
* [PyTorch](https://www.pytorch.org/): as the main ML framework
* [Comet.ml](https://www.comet.ml): tracking code, logging experiments
* [OmegaConf](https://omegaconf.readthedocs.io/en/latest/): for managing configuration files

First create a virtual env for the project. 
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the latest version of PyTorch from the [official site](https://www.pytorch.org/). Finally, run the following:
```bash
pip install -r requirements.txt
```

## Usage

To use this code, follow these steps:
+ Create data to work with by running the data_processing code.
+ On every code, please specify the "data_name" variable to be 'ML1M'/'Yahoo'/'Pinterest', and the "recommender_name" variable to be 'MLP'/'VAE' or pass it through arguments of "recommender" and "data"

## Reproducing the Results:
+ After running the preprocessing step, simply run the recommenders_training.py and specify the "data_name" variable to be 'ML1M'/'Yahoo'/'Pinterest', and the "recommender_name" variable to be 'MLP'/'VAE'.
+ From the output checkpoints check which recommenders you want to pick for explanation. Then set the file name of the checkpoint in LXR_training.py or pass it as a argument by --directory and run to train the explainers. 
+ Then to get other explainers and evaluate LXR evaluation, run the metrics.py file. This will print all the numbers you want. We have all these outputs in "Experiments Results" folder.

## Results

#### Top-K recommenders on metric consistency
Comparison of CE methods based on POS@5 (lower value is the better) across 4 performance levels of the VAE recommender on ML-1M dataset. The figure shows the impact of going beyond Top-1 (a) and considering Top-k (b-d) recommendations on improving consistency when evaluating CE models. To facilitate clearer comparisons, the values are normalized using Min-max normalization, and shading is used to represent the variance in the results.
![RecLengthFig](https://github.com/dbis-uibk/CE4RS-Eval/blob/main/Results/Figures/TopK.png)

#### Explanation Sparsity Metric
Performance of CE methods across three datasets based on explanation sparsity metric using VAE recommender. The evaluation is conducted over eight explanation sizes, providing a comparative analysis of the methods. To facilitate clearer comparisons, the values are normalized using Min-max normalization. The results highlight dataset-specific performance variations, reflecting the effectiveness of each CE method on specific sparsity levels.
![XpSizeFig](https://github.com/dbis-uibk/CE4RS-Eval/blob/main/Results/Figures/xpSize.png)

#### Metric Consistency on MF Recommender
![TopKMFRec](https://github.com/dbis-uibk/CE4RS-Eval/blob/main/Results/Figures/TopkMF.png)

#### POS consistency on Pinterest dataset
![TopKpinterestVAE](https://github.com/dbis-uibk/CE4RS-Eval/blob/main/Results/Figures/TopkVAEPinterest.png)

#### Consistency Evaluation effects of Top1 to Top5 
![TopKExcel](https://github.com/dbis-uibk/CE4RS-Eval/blob/main/Results/Figures/TopKinExcel.png)

#### POS@20 on ML1M dataset and VAE Recommender
 ![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Results/Figures/pos20ML1MVAE.png)

#### POS@10 on Yahoo dataset and  MF Recommender
![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Results/Figures/POSYahooMF.png)

#### POS@10 MF Recommender and  Pinterest dataset
![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Results/Figures/POS@10MFPinterest.png)

#### Evaluation based on ONLY Top-1
![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/MLP%20ML-1M.png)


## Acknowledgements
Thanks to [LXR] for making their code public.

## Citation
If you find the code helpful, please cite this work:
```

```

