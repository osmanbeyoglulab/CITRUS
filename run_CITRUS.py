#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Yifeng Tao", "Xiaojun Ma"
# Last update: March 2021
# =============================================================================
""" 
Demo of training and evaluating CITRUS model and its variants.

"""
import os
import argparse
from utils import bool_ext, load_dataset, split_dataset, evaluate, checkCorrelations
from models import CITRUS
import pickle


parser = argparse.ArgumentParser()

parser.add_argument(
    "--train_model",
    help="whether to train model or load model",
    type=bool_ext,
    default=True,
)
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
    default=1000
)
parser.add_argument(
    "--max_fscore",
    help="Max F1 score to early stop model from training",
    type=float,
    default=0.7
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
    "--test_inc_size",
    help="increment interval size between log outputs",
    type=int,
    default=256
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
    "--mask01",
    help="wether to ignore the float value and convert mask to 01",
    type=bool_ext,
    default=True,
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
)

args = parser.parse_args()


print("Loading dataset...")
dataset, dataset_test = load_dataset(
    input_dir=args.input_dir,
    mask01=args.mask01,
    dataset_name=args.dataset_name,
    deg_normalization=args.deg_normalization,
)


train_set, test_set = split_dataset(dataset, ratio=0.66)

args.can_size = dataset["can"].max()  # cancer type dimension
args.sga_size = dataset["sga"].max()  # SGA dimension
args.deg_size = dataset["deg"].shape[1]  # DEG output dimension
args.num_max_sga = dataset["sga"].shape[1]  # maximum number of SGAs in a tumor

args.hidden_size = dataset["tf_gene"].shape[0]
print("Hyperparameters:")
print(args)
args.tf_gene = dataset["tf_gene"]

model = CITRUS(args)  # initialize CITRUS model
model.build()  # build CITRUS model
# model.cuda()
if args.train_model:  # train from scratch
    print("Training...")
    model.fit(
        train_set,
        test_set,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        max_iter=args.max_iter,
        max_fscore=args.max_fscore,
        test_inc_size=args.test_inc_size,
    )
    model.load_model(os.path.join(args.output_dir, "trained_model.pth"))
else:  # or directly load trained model
    model.load_model(os.path.join(args.input_dir, "trained_model.pth"))
    
# evaluation
print("Evaluating...")
labels, preds, _, _, _, _, _ = model.test(
    test_set, test_batch_size=args.test_batch_size
)
print("Performance on validation set:\n")
checkCorrelations(labels, preds)

# get training attn_wt and others
labels, preds, hid_tmr, emb_tmr, emb_sga, attn_wt, tmr = model.test(
    dataset, test_batch_size=args.test_batch_size
)

# predict on holdout and evaluate the performance
labels_test, preds_test, _, _, _, _, tmr_test = model.test(dataset_test, test_batch_size=args.test_batch_size)
print("Performance on holdout test set:\n")
checkCorrelations(labels_test, preds_test)

# get gene emb
gene_emb = model.layer_sga_emb.weight.data.cpu().numpy()

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

with open(os.path.join(args.output_dir, "output_{}_{}.pkl".format(args.dataset_name, args.tag)), "wb") as f:
  pickle.dump(dataset_out, f, protocol=2)
