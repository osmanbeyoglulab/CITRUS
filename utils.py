#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Yifeng Tao", "Xiaojun Ma"
# Last update: March 2021
# =============================================================================
""" 
Shared utilities for models.py and test_run.py.

"""
import os
import random
import numpy as np
import pickle

import torch
from torch.autograd import Variable

from scipy import stats
from sklearn.preprocessing import normalize, scale
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

import torch.nn as nn




def bool_ext(rbool):
    """Solve the problem that raw bool type is always True.

    Parameters
    ----------
    rbool: str
      should be True of False.

    """

    if rbool not in ["True", "False"]:
        raise ValueError("Not a valid boolean string")
    return rbool == "True"


def load_dataset(
    input_dir="data", mask01=False, dataset_name="", gep_normalization="scaleRow"
):

    # load dataset
    data = pickle.load(
        open(os.path.join(input_dir, "{}.pkl".format(dataset_name)), "rb")
    )
    can_r = data["can"]  # cancer type index of tumors: list of int
    sga_r = data["sga"]  # SGA index of tumors: list of list
    gep = data["gep"]  # gep matrix of tumors: continuous data
    tmr = data["tmr"]  # barcodes of tumors: list of str
    tf_gene = np.array(data["tf_gene"])

    #load holdout dataset
    can_r_test = data["can_test"] # cancer type index of tumors: list of int
    sga_r_test = data["sga_test"] # SGA index of tumors: list of list
    gep_test = data["gep_test"]   # GEP matrix of tumors: continuous data
    tmr_test = data["tmr_test"]   # barcodes of tumors: list of str

    if mask01 == True:
        tf_gene[tf_gene != 0] = 1
    else:
        tf_gene = normalize(tf_gene)

    # shift the index of cancer type by +1, 0 is for padding
    can = np.asarray([[x + 1] for x in can_r], dtype=int)

    # shift the index of SGAs by +1, 0 is for padding
    num_max_sga_train = max([len(s) for s in sga_r])
    num_max_sga_test = max([len(s) for s in sga_r_test])
    num_max_sga = max(num_max_sga_train, num_max_sga_test)
    sga = np.zeros( (len(sga_r), num_max_sga), dtype=int )
    for idx, line in enumerate(sga_r):
        line = [s+1 for s in line]
        sga[idx,0:len(line)] = line

    
    if gep_normalization == "scaleRow":
        gep = scale(gep, axis=1)


    # shift the index of cancer type by +1, 0 is for padding
    can_test = np.asarray([[x+1] for x in can_r_test], dtype=int)

    # shift the index of SGAs by +1, 0 is for padding

    sga_test = np.zeros( (len(sga_r_test), num_max_sga), dtype=int )
    for idx, line in enumerate(sga_r_test):
        line = [s+1 for s in line]
        sga_test[idx,0:len(line)] = line  
  
    if gep_normalization == 'scaleRow':
        gep_test = scale(gep_test, axis = 1)   
  
    dataset = {"can":can, "sga":sga, "gep":gep, "tmr":tmr, "tf_gene":tf_gene}
    dataset_test = {"can":can_test, "sga":sga_test, "gep":gep_test, "tmr":tmr_test}
    
    return dataset, dataset_test

    
def split_dataset(dataset, ratio=0.66):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.34)  # , random_state=2020)
    X = dataset["sga"]
    y = dataset["can"]

    train_set = {}
    test_set = {}
    for train_index, test_index in sss.split(X, y):  # it only contains one element

        train_set = {
            "sga": dataset["sga"][train_index],
            "can": dataset["can"][train_index],
            "gep": dataset["gep"][train_index],
            "tmr": [dataset["tmr"][idx] for idx in train_index],
        }
        test_set = {
            "sga": dataset["sga"][test_index],
            "can": dataset["can"][test_index],
            "gep": dataset["gep"][test_index],
            "tmr": [dataset["tmr"][idx] for idx in test_index],
        }
    return train_set, test_set


def shuffle_data(dataset):
    rng = list(range(len(dataset["can"])))
    random.Random(2020).shuffle(rng)
    dataset["can"] = dataset["can"][rng]
    dataset["sga"] = dataset["sga"][rng]
    dataset["gep"] = dataset["gep"][rng]
    dataset["tmr"] = [dataset["tmr"][idx] for idx in rng]
    return dataset


def wrap_dataset(dataset):
    """Wrap default numpy or list data into PyTorch variables."""
    dataset["can"] = Variable(torch.LongTensor(dataset["can"]))
    dataset["sga"] = Variable(torch.LongTensor(dataset["sga"]))
    dataset["gep"] = Variable(torch.FloatTensor(dataset["gep"]))

    return dataset


def get_minibatch(dataset, index, batch_size, batch_type="train"):
    """Get a mini-batch dataset for training or test.

    Parameters
    ----------
    dataset: dict
      dict of lists, including SGAs, cancer types, GEPs, patient barcodes
    index: int
      starting index of current mini-batch
    batch_size: int
    batch_type: str
      batch strategy is slightly different for training and test
      "train": will return to beginning of the queue when `index` out of range
      "test": will not return to beginning of the queue when `index` out of range

    Returns
    -------
    batch_dataset: dict
      a mini-batch of the input `dataset`.

    """

    sga = dataset["sga"]
    can = dataset["can"]
    gep = dataset["gep"]
    tmr = dataset["tmr"]

    if batch_type == "train":
        batch_sga = [sga[idx % len(sga)] for idx in range(index, index + batch_size)]
        batch_can = [can[idx % len(can)] for idx in range(index, index + batch_size)]
        batch_gep = [gep[idx % len(gep)] for idx in range(index, index + batch_size)]
        batch_tmr = [tmr[idx % len(tmr)] for idx in range(index, index + batch_size)]
    elif batch_type == "test":
        batch_sga = sga[index : index + batch_size]
        batch_can = can[index : index + batch_size]
        batch_gep = gep[index : index + batch_size]
        batch_tmr = tmr[index : index + batch_size]
    batch_dataset_in = {
        "sga": batch_sga,
        "can": batch_can,
        "gep": batch_gep,
        "tmr": batch_tmr,
    }

    batch_dataset = wrap_dataset(batch_dataset_in)
    return batch_dataset


def evaluate(labels, preds, epsilon=1e-4):
    """Calculate performance metrics given ground truths and prediction results.

    Parameters
    ----------
    labels: matrix of 0/1
      ground truth labels
    preds: matrix of float in [0,1]
      predicted labels
    epsilon: float
      a small Laplacian smoothing term to avoid zero denominator

    Returns
    -------
    precision: float
    recall: float
    f1score: float
    accuracy: float

    """

    flat_labels = np.reshape(labels, -1)
    flat_preds = np.reshape(preds, -1)

    corr_spearman = stats.spearmanr(flat_preds, flat_labels)[0]
    corr_pearson = stats.pearsonr(flat_preds, flat_labels)[0]
    return (corr_spearman, corr_pearson)


def checkCorrelations(labels, preds):
    corr_row_pearson = 0
    corr_row_spearman = 0
    corr_col_pearson = 0
    corr_col_spearman = 0
    nsample = labels.shape[0]
    ngene = labels.shape[1]

    for i in range(nsample):
        corr_row_pearson += stats.pearsonr(preds[i, :], labels[i, :])[0]
        corr_row_spearman += stats.spearmanr(preds[i, :], labels[i, :])[0]
    corr_row_pearson = corr_row_pearson / nsample
    corr_row_spearman = corr_row_spearman / nsample

    print(
        "spearman sample mean: %.3f, pearson sample mean: %.3f"
        % (corr_row_spearman, corr_row_pearson)
    )

    for j in range(ngene):
        corr_col_pearson += stats.pearsonr(preds[:, j], labels[:, j])[0]
        corr_col_spearman += stats.spearmanr(preds[:, j], labels[:, j])[0]
    corr_col_pearson = corr_col_pearson / ngene
    corr_col_spearman = corr_col_spearman / ngene

    print(
        "spearman gene mean: %.3f, pearson gene mean: %.3f"
        % (corr_col_spearman, corr_col_pearson)
    )


class EarlyStopping(object):
    def __init__(self, mode="max", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)
