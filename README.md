# Introduction

The repository contains the implementation in PyTorch of CITRUS model in the paper -----. CITRUS(Chromatin-informed Inference of Transcriptional Regulators Using Self-attention mechanism) is a partially interpretable deep neural network modeling the impact of somatic alterations on cellular states onto downstream gene expression programs.The model uses a layer of hidden nodes to explicitly represent the state of transcription factors (TFs)

## Data
We have packaged all required data into a .pkl file for fast processing and ease of use. The file include Somatica alteration gene matrix, gene expression matrix, TF and target gene corresponding matrix, tumor and cancer types information which can be downloaded from : ......

## Prerequisites
The code runs on python 3.7 and above. Besides python 3, some other packages such as PyTorch, Pandas, Numpy, scikit-learn, Scipy are used. We have tested our code on torch verion 1.2.0 (Windows), torch version 1.5.1+cu101 (Linus), torch version .....(Max)

It is recomended to installl PyTorch through Anaconda package manager, since it install all denpendencies. If you installed the Anaconda distribution of Python 3.7+, Pandas, Numpy, scikit-learn, Scipy come pre-installed and no further installation steps are necessary.

To setup running environment, here are the easy steps to follow.

## In 
Download the reporsitory from https://github.com/osmanbeyoglulab/SPaRTAN

Install python3.7 or later. pySpaRTAN used the following dependencies as well: pandas, numpy, scipy, sklearn, matplotlib. 

You can install python dependencies by the following commands:
```sh
pip install pandas
pip install numpy
pip install scipy
pip install -U scikit-learn
pip install matplotlib
```
Cython is not reqired to be installed unless the pre-built Cython extensions do not work on your system. 

Our codes have been tested on Linux, Mac, and Windows systems. Please see Prerequisites.xlsx for the version of packages we tested on each operating system.

### Cython extension compilation

If the Cython extensions, which are platform-dependent binary modules, are not compatible with your operating system, additional built of those Cython extensions are needed on your own machine. 

First, install Cython by running the command
```sh
pip install "cython>0.21"    
```
Cython requires a C compiler to be present on the system. Please check [here](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) for C compiler installation on various operating systems

After installing Cython and C compiler, navigate to the directory pySPaRTAN and type in the command:
```sh
python setup.py build_ext --inplace
```
This will generate new Cython extension .so files (or .pyd files on Windows). The previously downloaded .so and .pyd files are renamed to "*_old.so" and "*_old.pyd" 

## Usage

pySPaRTAN module has 3 input datasets, and 2 generated ouputs

**Input datasets**
```sh
D: dataframe of shape (N, Q)
   The dataframe with N genes and Q TFs
   
P: dataframe of shape (M, S)
   The dataframe with M cells and S proteins  
   
Y: dataframe of shape (N, M)
   The dataframe with N genes and M cells    
```
**Outputs**
```sh
projD: dataframe of shape (Q, M) 
       projected TF activities with Q TFs and M cells 
       
projP: dataframe of shape (S, M)
      projected protein expression with S proteins and M cells  
```

**Hyperparameters**

pySPaRTAN model has 2 hyperparameters: lamda and rsL2

    lamda : float > 0, default=0.001
            LASSO regularization for linear regression 
   
    rsL2 : float > 0, default=0.001
           ridge regularization for linear regression

We can run pySPaRTAN by giving some specific values to those hyperparameters or using default settings in the script.
We can also use cross-validation to find the optional values of those hyperparameters for the input datasets, and then run pySPaRTAN to generate the projected outputs.

**Command line**

To run pySPaRTAN module with default parameters, simply execute the following command to generate outputs 
```sh
python run_pySPaRTAN.py
```

To check all optional arguments, type in the command line
```sh
python run_pySPaRTAN.py -h
```
It shows each argument and its description:

    optional arguments:
      
      --input_dir INPUT_DIR
                             string, default='../data/inputs'
                             Directory of input files
      --output_dir OUTPUT_DIR
                             string, default: '../data/outputs'
                             Directory of output files
      --dataset_D DATASET_D
                             string, default='Dpbmc'
                             File name of (gene X TF) dataframe.
                             Requires .csv format,
                             only contains file name, not include '.csv' extension
      --dataset_P DATASET_P
                             string, default='Ppbmc5kn_CD16'
                             File name of (cell X protein) dataframe.
                             Requires .csv format,
                             only contains file name, not include '.csv' extension
      --dataset_Y DATASET_Y
                             string, default='Ypbmc5kn_CD16'
                             File name of (gene X cell) dataframe.
                             Requires .csv format,
                             only contains file name, not include '.csv' extension
      --lamda LAMDA         
                             float, value>0, default=0.001
                             LASSO regularization for linear regression.
      --rsL2 RSL2           
                             float, value>0, default=0.001
                             Ridge regularization for linear regression,
      --normalization NORMALIZATION
                             string, default='l2'
                             Type of normalization performed on matrices,
                             if set to empty(''), then no normalization
      --fold FOLD           
                             int, value>=0, default=0
                             How many folds for the cross_validation.
                             value=0 means no cross_validation and using default/specified parameters
      --correlation
                             string, ['pearson', 'spearman'], default='pearson'
                             Type of correlation coefficient
                             
For example, we can use the following command to run the pySPaRTAN model with 5-fold cross-validation and using spearman correlation:

    python run_pySPaRTAN.py --fold 5 --correlation spearman


