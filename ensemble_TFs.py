#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Xiaojun Ma
# Created Date: Sep. 7, 2021
# =============================================================================
"""
This script generates the ensembled TF activities from multiple CITRUS training outputs

"""
 
import pickle
import pandas as pd 
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir", 
    help="directory of input files", 
    type=str, 
    default="./data"
)
parser.add_argument(
    "--output_dir",
    help="directory of output files",
    type=str,
    default="./output",
)
parser.add_argument(
    "--runs",
    help="number of runs",
    type=int,
    default=10,
)
args = parser.parse_args()

def generateTFactivity(tf, idx2can, tmr, cans, tf_name):
    # generate the TF activity matrix
    df_TF = pd.DataFrame(data = tf, columns = tf_name, index = tmr)
    can_names = [idx2can[idx] for idx in cans]
    df_TF["cancer_type"] = can_names
    return(df_TF)

#readin input dataset
data = pickle.load( open(os.path.join(args.input_dir,"dataset_CITRUS.pkl"), "rb"))

# read in output datasets of 10 runs
Ntf = args.runs
run = list()
for i in range(1,Ntf+1):
    dataset = pickle.load( open(os.path.join(args.output_dir,"output_dataset_CITRUS_{}.pkl".format(i)), "rb") )
    run.append(dataset) 

tfs = list()
for i in range(Ntf):
    tfs.append(run[i]["hid_tmr"])

## genereate ensemble tf activity matrix
tf_ensemble = 0
for i in range(Ntf):
    tf_ensemble += tfs[i]
    
tf_ensemble = tf_ensemble/Ntf

df_tf = generateTFactivity(tf_ensemble, data["idx2can"],data["tmr"], data["can"], data["tf_name"])

#save to file
df_tf.to_csv(os.path.join(args.output_dir,"TF_activity_ensemble_{}.csv".format(Ntf)))
