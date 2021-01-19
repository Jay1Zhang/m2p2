#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch


FOLDS_DIR = './folds_split/'    # folds directory
META_FILE = './qps_index.csv'   # meta-data file

# employed hyper-parameters & constants
BATCH = 32
N_WORKERS = 4   # thread number of dataloader
N_FEATS = 16
#MAX_DUR = 482   # we divided all speaking length by this max length to normalize

#UPD_WEIGHT = 0.5 # update rate for modality weights
#BETA = 50 # weight in the softmax function for modality weights

N_EPOCHS = 200  # master training procedure (alg 1 in paper)
n_EPOCHS = 10   # slave training procedure (alg 1 in paper)

# optimizer
LR = 1e-3
W_DECAY = 1e-5

# Device configuration
# questions here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)

