#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse

import utils, model
from dataset import gen_dataloader

if __name__ == '__main__':
    # 0 - Configure arguments parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--fd', required=False, default=0, type=int, help='fold id')
    parser.add_argument('--mod', required=False, default='avl', type=str,
                        help='modalities: a,v,l, or any combination of them')
    parser.add_argument('--dp', required=False, default=0.4, type=float, help='dropout')

    ## boolean flags
    parser.add_argument('--test_mode', default=False, action='store_true',
                        help='test mode: loading a pre-trained model and calculate loss')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='print more information')

    args = parser.parse_args()
    FOLD = int(args.fd)     # fold id
    MODS = list(args.mod)   # modalities: a, v, l
    DP = args.dp

    TEST_MODE = args.test_mode
    VERBOSE = args.verbose

    #############

    # 1 - load dataset
    tra_loader, val_loader, tes_loader = gen_dataloader(FOLD, MODS)

    # 2 - Initialize m2p2 models
    # initialize multiple models to output the latent embeddings for a,v,l
    latent_models = {mod: model.LatentEmb(mod, utils.N_FEATS, utils.N_FEATS//2, DP) for mod in MODS}
    # initialize the shared mlp model for alignment module
    shared_mlp_model = model.AlignEmb(ninp=utils.N_FEATS, nout=utils.N_FEATS, dropout=DP)
    # initialize the reference mlp model for heterogeneity module
    # Note: test_mode doesn't need reference model because it's used to update params: w_m
    if not TEST_MODE:
        ref_mlp_models =
    # initialize persuasiveness model to predict persuasiveness with H_align, H_het and X_meta

    # 3 - Initialize trained parameters: theta and concat weights: w_a, w_v, w_l

    # 4 - Train or Test
    if not TEST_MODE:
        #### Master Procedure Start ####

        #### Slave Procedure Start ####

        #### Slave Procedure End ####

        #### Master Procedure End ####
        pass
    else:
        pass
