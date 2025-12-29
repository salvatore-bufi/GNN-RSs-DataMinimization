# Resilience and Effectiveness of GNN-based Recommender Systems under Data Minimization

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
  - [Installation guidelines: scenario](#installation-guidelines-scenario)
- [Datasets](#datasets)
- [Recommendation Lists](#recommendation-lists)
- [Usage](#usage)
  - [Reproduce Paper Results](#reproduce-paper-results)


## Requirements 

This software has been executed on the operative system Ubuntu `20.04`.

Please, make sure to have the following installed on your system:

* Python `3.8.0`
* PyTorch Geometric with PyTorch `2.1.2` 
* CUDA `12.1`

### Installation guidelines: scenario
If you have the possibility to install CUDA on your workstation (i.e., `12.1`), you may create the virtual environment with the requirements files we included in the repository, as follows:

```
# PYTORCH ENVIRONMENT 

$ python3.8 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

## Model Training and Evaluation

The code in this repository allows replicating the experimental setting described within the paper.

The recommenders training and evaluation procedures have been developed on the reproducibility framework **Elliot**,
so we suggest you refer to the official GitHub 
[page](https://github.com/sisinflab/elliot) and 
[documentation](https://elliot.readthedocs.io/en/latest/).

Regarding the graph-based recommendation models based on torch, they have been implemented
in `PyTorch Geometric`, with `PyTorch` `2.1.2` and CUDA `12.1`.

For your convenience, we provide the files containing each model performance at `./data/` when they are trained with minimized dataset and subsampled dataset.

## Regression Training

At `./data/` you will find containing each model performance when they are trained with minimized dataset and subsampled dataset. These files contain also the dataset characterisitcs.

To train the regression and gather the paper result, you may run the following commands for the data minimization and subsamples scenarios, respectively.

```
python regression_minimzation.py
```

```
python regression_subsample.py
```



