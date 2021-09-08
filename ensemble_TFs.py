#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Xiaojun Ma
# Created Date: Sep 7 PDT 2021
# =============================================================================
"""
This script generates the ensembled TF activities from multiple CITRUS training outputs

"""
 
import pickle
import pandas as pd 
import os

# input_dir = "./data"
# output_dir = "./data"

input_dir = "./data"
output_dir = "./data"

def generateTFactivity(tf, idx2can, tmr, cans, tf_name):
    # generate the TF activity matrix
    df_TF = pd.DataFrame(data = tf, columns = tf_name, index = tmr)
    can_names = [idx2can[idx] for idx in cans]
    df_TF['cancer_type'] = can_names
    return(df_TF)

#readin input dataset
data = pickle.load( open(os.path.join(input_dir,"dataset_PANunion2500_17_sga_dropped_seperated_rmNotImpt_0.04_with_holdout_new.pkl"), "rb"))

# read in output datasets of 10 runs
Ntf = 2
run = list()
for i in range(1,Ntf+1):
    dataset = pickle.load( open(os.path.join(input_dir,"output_with_holdout_new_mask_batch100_inputdrop0.2_drop0.2_new_{}.pkl".format(i)), "rb") )

    run.append(dataset) 

tf = list()
for i in range(Ntf):
    tf.append(run[i]['hid_tmr'])

## genereate ensemble tf activity matrix
tf_ensemble = 0
for i in range(Ntf):
    tf_ensemble += tf[i]
    
tf_ensemble = tf_ensemble/Ntf

df_tf = generateTFactivity(tf_ensemble, data['idx2can'],data['tmr'], data['can'], data['tf_name'])

#save to file
df_tf.to_csv(os.path.join(output_dir,"TF_activity_ensemble_{}.csv".format(Ntf)))
