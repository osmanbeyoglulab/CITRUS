# Introduction

The repository contains the PyTorch implementation of CITRUS model in the paper -----. CITRUS(Chromatin-informed Inference of Transcriptional Regulators Using Self-attention mechanism) is a partially interpretable deep neural network modeling the impact of somatic alterations on cellular states onto downstream gene expression programs. The model uses a layer of hidden nodes to explicitly represent the state of transcription factors (TFs)

## Data
There are 3 major datasets needed to run CITRUS model: Somatica alteration gene matrix(SGA), gene expression matrix (GExp),  and TF-target gene matrix. Our SGA and GExp come from 5803 TCGA samples including 17 cancer types. We transformed the sparse binary SGA to index lists to facilitate gene embedding. We packaged those matrices as well as cancer type, tumor barcode of each sample into a pickle file for fast processing and convenient delivery. 
Data can be downloaded from ....

## Prerequisites
The code runs on python 3.7 and above. Besides python 3, some other packages such as PyTorch, Pandas, Numpy, scikit-learn, Scipy are used. We have tested our code on torch verion 1.2.0 (Windows), torch version 1.5.1+cu101 (Linus), torch version .....(Max)

It is recomended to installl PyTorch through Anaconda package manager since it installs all dependencies. If you installed the Anaconda distribution of Python 3.7 and above, Pandas, Numpy, scikit-learn, Scipy come pre-installed and no further installation steps are necessary.

To setup running environment, here are the easy steps to follow.
1. Install Anaconda: Download the Anaconda installer according to your operating system, and follow the installer's prompt to finish the installation. Please check Anaconda documentation here
https://docs.anaconda.com/anaconda/install/index.html
2. Create an conda environment and activate it by executing
```sh
    conda create --name myenv
    conda activate myenv
```
3. Install PyTorch in the conda enviroment. The install command can be automatically generated based on your computer hardware configuration by visiting PyTorch support site:https://pytorch.org/get-started/locally/. For example, on a Window system with CUDA support, you may install PyTorch by running
```sh
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
 ```  
 ## Replicate experiments
Download the data into ./data directory
Execute the command
```sh
python test_run.py > run.log
```sh

After training is down. The model evaluates the performance by checking the correlation betweenbetween overserved gene expression and predicted gene expression on the test set which is not shown in the training process. You can view the results in the run.log file
