## Introduction

The repository contains the PyTorch implementation of CITRUS model in the paper -----. 

CITRUS(Chromatin-informed Inference of Transcriptional Regulators Using Self-attention mechanism) is a partially interpretable deep neural network modeling the impact of somatic alterations on cellular states onto downstream gene expression patterns within context-speciﬁc transcriptional programs.  The model follows an overall encoder-decoder architecture, while the encoder module employs a self-attention mechanism to model the contextual functional impact of somatic alterations in a tumor-specific manner and the decoder uses a layer of hidden nodes to explicitly represent the state of transcription factors (TFs).

## Data
There are three major datasets used to train CITRUS model: Somatica alteration gene matrix(SGA), gene expression matrix (EPA),  and TF-target gene matrix. They included 5803 samples with seventeen different tumor types from TCGA. SGA is originally a binary matrix containing 11998 genes with 1 as having distinct mutation or copy number alteration and 0 as none of them. We transformed the sparse binary SGA into index lists to facilitate gene embedding. EPA contains continuous gene expression with filtered 5541 genes. We first select top 2500 variant genes for each cancer type and union all genes from all cancer types, and then intersect with the genes appeard in TF-target gene profile.  We packaged those matrices as well as cancer type, tumor barcode of each sample into a pickle file for fast processing and convenient delivery. 
    
The integrated dataset of the pikle file can be downloaded from https://sites.pitt.edu/~xim33/CITRUS

You can explore the contents of the dataset by running:
```sh
data = pickle.load( open("dataset_CITRUS.pkl"), "rb")
data.keys()
```

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

We use the ensemble method to further stabilize TF activity which is a hidden layer before the output gene expression layer.

To generate ensembled TF activity, First run the model 10 times
```sh
for (( i = 1; N <= 10; i++ ))
    python test.py --tag i
done
```
After getting the outputs for all runs, execute the TF_ensemble.py to ensemble TF activities
```sh
python TF_ensemble.py --runs 10
```
We also have uploaded ensembled TF activities generated from 10 runs of CITURS model to the website https://sites.pitt.edu/~xim33/CITRUS

## Explore CITRUS model

**Arguments ans Hyperparameters**

To assist CITRUS usage we established the following input arguments

```sh
parser.add_argument(
    "--input_dir",
    help="directory of input files",
    type=str,
    default="../data"
)
parser.add_argument(
    "--output_dir",
    help="directory of output files",
    type=str,
    default="./output",
)
parser.add_argument(
    "--dataset_name",
    help="the dataset name loaded and saved",
    type=str,
    default="dataset_PANunion2500_17_sga_dropped_seperated_rmNotImpt_0.04_with_holdout_new",
)
parser.add_argument(
    "--tag",
    help="a tag passed from command line",
    type=str,
    default=""
)
parser.add_argument(
    "--run_count",
    help="the count for training",
    type=str,
    default="1"
parser.add_argument(
    "--train_model",
    help="whether to train model or load model",
    type=bool_ext,
    default=True,
)
```
For example, The following command runs CITIRUS by specifying your own dataset and its location:
```sh
python test_run.py --dataset_name "mydataset" --input_dir "path/to/data"
```
All the demos we showed above trained CITRUS deep learning model with default hyperparameters. There are more than 10 hyperparameters that have been tuned to get the optimal results for this dataset.

```sh
parser.add_argument(
    "--embedding_size",
    help="embedding dimension of genes and tumors",
    type=int,
    default=512,
)
parser.add_argument(
    "--hidden_size",
    help="hidden layer dimension of MLP decoder",
    type=int,
    default=400
)
parser.add_argument(
    "--attention_size",
    help="size of attention parameter beta_j",
    type=int,
    default=256
)
parser.add_argument(
    "--attention_head",
    help="number of attention heads",
    type=int,
    default=32
)
parser.add_argument(
    "--learning_rate",
    help="learning rate for Adam",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--max_iter",
    help="maximum number of training iterations",
    type=int,
    default=1
)
parser.add_argument(
    "--batch_size",
    help="training batch size",
    type=int,
    default=100
)
parser.add_argument(
    "--test_batch_size",
    help="test batch size",
    type=int,
    default=100
)
parser.add_argument(
    "--dropout_rate",
    help="dropout rate",
    type=float,
    default=0.2
)
parser.add_argument(
    "--input_dropout_rate",
    help="dropout rate",
    type=float,
    default=0.2
)
parser.add_argument(
    "--weight_decay",
    help="coefficient of l2 regularizer",
    type=float,
    default=1e-5
)
parser.add_argument(
    "--activation",
    help="activation function used in hidden layer",
    type=str,
    default="tanh",
)
parser.add_argument(
    "--patience",
    help="earlystopping patience",
    type=int,
    default=30
)
parser.add_argument(
    "--deg_normalization",
    help="how to normalize deg",
    type=str,
    default="scaleRow"
)
parser.add_argument(
    "--attention",
    help="whether to use attention mechanism or not",
    type=bool_ext,
    default=True,
)
parser.add_argument(
    "--cancer_type",
    help="whether to use cancer type or not",
    type=bool_ext,
    default=True,
)

```
**Extract contents from CITRUS output**
The output contains many elements including "predicted gene expression", "extracted TF activities", "gene embedding", "tumor embedding", "attention weight", etc. They are integrated into a python dictionary data structure as:

The output dictory object was serialized to a byte stream and saved as a pickle file on disk. 
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

To extract the contents of output. First readin the pickle file
```sh
# suppose the output file in ./data as well
output_data = pickle.load(open("./data/output_dataset_CITRUS.pkl", "rb"))
```

First, need to readin the pickle file.
```sh
# suppose the output file in ./data as well
output_data = pickle.load(open("./data/output_dataset_CITRUS.pkl", "rb"))
```
Then extract the elements, here are a few demos:
```sh
gene_prediction = output_data['preds']
attention_weight = output_data['attn_wt']
gene_embedding= output_data['gene_emb']
tumor_embedding = output_data['emb_tmr'
TF_activity = output_data['hid_tmr']
```


