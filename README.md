# Introduction

The repository contains the PyTorch implementation of CITRUS model in the paper -----. 

CITRUS(Chromatin-informed Inference of Transcriptional Regulators Using Self-attention mechanism) is a partially interpretable deep neural network modeling the impact of somatic alterations on cellular states onto downstream gene expression patterns within context-speciﬁc transcriptional programs.  The model follows an overall encoder-decoder architecture, while the encoder module employs a self-attention mechanism to model the contextual functional impact of somatic alterations in a tumor-specific manner and the decoder uses a layer of hidden nodes to explicitly represent the state of transcription factors (TFs).

## Data
There are 3 major datasets for CITRUS model: Somatica alteration gene matrix(SGA), gene expression matrix (GExp),  and TF-target gene matrix. Our SGA and GExp come from 5803 TCGA samples including 17 cancer types. We transformed the sparse binary SGA to index lists to facilitate gene embedding. We packaged those matrices as well as cancer type, tumor barcode of each sample into a pickle file for fast processing and convenient delivery. 
data which can be downloaded from ....

## Prerequisites
The code runs on python 3.7 and above. Besides python 3, some other packages such as PyTorch, Pandas, Numpy, scikit-learn, Scipy are used. We have tested our code on torch verion 1.2.0 (Windows), torch version 1.5.1+cu101 (Linus), torch version .....(Mac)

It is recomended to installl PyTorch through Anaconda package manager since it installs all dependencies. If you installed the Anaconda distribution of Python 3.7 and above, Pandas, Numpy, scikit-learn, Scipy come pre-installed and no further installation steps are necessary.

To setup running environment, here are the easy steps to follow.
1. Install Anaconda: Download the Anaconda installer according to your operating system, and follow the installer's prompt to finish the installation. Please check Anaconda documentation here
https://docs.anaconda.com/anaconda/install/index.html
2. Create an conda environment and activate it by executing
```sh
    conda create --name myenv
    conda activate myenv
```
3.  Install PyTorch in the conda enviroment. The installation command will be slightly different depending on your computer operating system and hardware configuration. You can get customized the installation command by visiting PyTorch support site:https://pytorch.org/get-started/locally/. For example, on a Window system with CUDA support, you may install PyTorch by running
```sh
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
 ```  
 ## Replicate experiments
Download the data into ./data directory and execute the command
```sh
python test_run.py 
```
The output is a dictionary packed in a pickle file, containing predicted gene expression, extracted TF activities, gene embedding, tumor embedding, etc.  
```sh
dataset_out = {
    "labels": labels,         # measured exp 
    "preds": preds,           # predicted exp
    "hid_tmr": hid_tmr,       # TF activity
    "emb_tmr": emb_tmr,       # tumor embedding
    "tmr": tmr,               # tumor list
    "emb_sga": emb_sga,       # straitified tumor embedding
    "attn_wt": attn_wt,       # attention weight
    "can": dataset["can"],    # cancer type list
    "gene_emb": gene_emb,     # gene embedding
    "tf_gene": model.layer_w_2.weight.data.cpu().numpy(),  # trained weight of tf_gene constrains
    'labels_test':labels_test,      # measured exp on test set
    'preds_test':preds_test,        # predicted exp on test set
    'tmr_test':tmr_test,            # tumor list on test set
    'can_test':dataset_test['can']  # cancer type list on test set
}
```

We use ensemble method to furthur stablize TF activity which is a hidden layer before the output gene expression layer. 
To generate ensembled TF activity, First run the model 10 times by execute the command
```sh
for (( i = 1; N <= 10; i++ ))
    python test.py --tag i 
done
```
After getting the output results of all runs, Then execute the following command to generate the ensembled TF activity
```sh
    python TF_ensemble.py
```

