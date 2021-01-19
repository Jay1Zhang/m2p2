#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse

if __name__ == '__main__':
    # 0 - Configure arguments parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--fd', required=False, default=0, type=int, help='fold id')
    parser.add_argument('--mod', required=False, default='avl', type=str,
                        help='modalities: a,v,l, or any combination of them')

    ## boolean flags
    parser.add_argument('--test_mode', default=False, action='store_true',
                        help='test mode: loading a pre-trained model and calculate loss')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='print more information')

    args = parser.parse_args()
    FOLD = int(args.fd)     # fold id
    MODS = list(args.mod)   # modalities: a, v, l

    TEST_MODE = args.test_mode
    VERBOSE = args.verbose

    #############

    # 1 - load dataset

    # 2 - Initialize m2p2 models
    ## initialize multiple model to output the latent embeddings for a,v,l
    mul_model = {}

    ## initialize the shared mlp model for alignment module

    ## initialize the reference mlp model for heterogeneity module
    ## Note: test_mode doesn't need reference model because it's used to update params: w_m

    ## initialize persuasiveness model to predict persuasiveness with H_align, H_het and X_meta

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
