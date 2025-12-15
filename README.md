# Legal but Unfair: Auditing the Impact of Data Minimization on Fairness and Accuracy Trade-off in Recommender Systems

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
  - [Installation guidelines: scenario](#installation-guidelines-scenario)
- [Datasets](#datasets)
- [Recommendation Lists](#recommendation-lists)
- [Usage](#usage)
  - [Reproduce Paper Results](#reproduce-paper-results)



## Description

The code in this repository allows replicating the experimental setting described within the paper.

The recommenders training and evaluation procedures have been developed on the reproducibility framework **Elliot**,
so we suggest you refer to the official GitHub 
[page](https://github.com/sisinflab/elliot) and 
[documentation](https://elliot.readthedocs.io/en/latest/).

Regarding the graph-based recommendation models based on torch, they have been implemented
in `PyTorch Geometric`, with `PyTorch` `2.1.2` and CUDA `12.1`.

For granting the usage of the same environment on different machines, 
all the experiments have been executed on the same docker container.
If the reader would like to use it, 
please look at the corresponding section in [requirements](#requirements).

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


## Datasets

At `./data/` you may find all the [files](data) related to 
the datasets

The datasets could be found within the directory `./data/[DATASET]/`. 


## Recommendation Lists

The best models recommendation lists could be found at `./results/[DATASET]_[MINIMIZATION_STRATEGY]_[n]/recs` once the experiments have been completed.
You may be use them for computing the recommendation metrics.


## Usage

Here we describe the steps to reproduce the results presented in the paper. 
Furthermore, we provide a description of how the experiments have been configured.

### Reproduce Paper Results

[Here](run_experiments.py) you can find a ready-to-run Python file with all the pre-configured experiments cited in our paper.
You can easily run them with the following command:

```
python run_experiments.py
```

It reproduces all experimental results from our paper. The script implements all baseline models across both datasets, testing each minimization strategy and interaction number combination.
The results will be stored in the folder ```results/[DATASET]_[MINIMIZATION_STRATEGY]_[n]```.



