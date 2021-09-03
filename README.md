# Introduction

The repository contains the implementation in PyTorch of CITRUS model in the paper -----. CITRUS(Chromatin-informed Inference of Transcriptional Regulators Using Self-attention mechanism) is a partially interpretable deep neural network modeling the impact of somatic alterations on cellular states onto downstream gene expression programs.The model uses a layer of hidden nodes to explicitly represent the state of transcription factors (TFs)

## Data
We have packaged all required data into a .pkl file for fast processing and ease of use. The file include Somatica alteration gene matrix, gene expression matrix, TF and target gene matrix, tumor and cancer types information. The data can be downloaded from : ......

## Prerequisites
The code runs on python 3.7 and above. Besides python 3, some other packages such as PyTorch, Pandas, Numpy, scikit-learn, Scipy are used. We have tested our code on torch verion 1.2.0 (Windows), torch version 1.5.1+cu101 (Linus), torch version .....(Max)

It is recomended to installl PyTorch through Anaconda package manager, since it install all denpendencies. If you installed the Anaconda distribution of Python 3.7+, Pandas, Numpy, scikit-learn, Scipy come pre-installed and no further installation steps are necessary.

To setup running environment, here are the easy steps to follow.
1. Install Anaconda: Download specific Anaconda installer according to your operating system, and follow the installer's prompt to finish installation. Please check Anaconda documentation here
https://docs.anaconda.com/anaconda/install/index.html
2. Create an conda environment and activate it by executing
    conda create --name myenv
    conda activate myenv
3. Install PyTorch in the conda enviroment. The install command can be automatially genereated based on your computer hardware configuration by visiting PyTorch support site:https://pytorch.org/get-started/locally/. For example, on Window system with CUDA support, you may install PyTorch by running
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    
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


