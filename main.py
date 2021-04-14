#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import time
from tqdm import tqdm
from torch import optim

import model
from utils import *
from dataset import gen_dataloader
from train import train_m2p2, eval_m2p2, train_ref, eval_ref

if __name__ == '__main__':
    # 0 - Configure arguments parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--fd', required=False, default=9, type=int, help='fold id')
    parser.add_argument('--mod', required=False, default='avl', type=str,
                        help='modalities: a,v,l, or any combination of them')
    parser.add_argument('--dp', required=False, default=0.2, type=float, help='dropout')

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
    #TEST_MODE = True
    #VERBOSE = True
    print("FOLD", FOLD)

    #############

    # 0 - Device configuration
    config_device()

    # 1 - load dataset
    tra_loader, val_loader, tes_loader = gen_dataloader(FOLD, MODS)

    # 2 - Initialize m2p2 models
    # initialize multiple models to output the latent embeddings for a,v,l
    latent_models = {mod: model.LatentModel(mod, N_FEATS, N_FEATS, DP).to(device) for mod in MODS}
    # initialize the shared mlp model for alignment module
    align_model = model.AlignModel(ninp=N_FEATS, nout=N_FEATS, dropout=DP).to(device)
    # initialize the reference mlp model for heterogeneity module
    # Note: test_mode doesn't need reference model because it's used to update params: w_m
    if not TEST_MODE:
        ref_model = model.HetModel(nfeat=N_FEATS, nhid=N_FEATS // 2, dropout=DP).to(device)
        ref_params = get_hyper_params({'ref': ref_model})
        ref_optim = optim.Adam(ref_params, lr=LR, weight_decay=W_DECAY)
        ref_scheduler = optim.lr_scheduler.StepLR(ref_optim, step_size=STEP_SIZE, gamma=SCHE_GAMMA)
    # initialize persuasiveness model to predict persuasiveness with H_align, H_het and X_meta
    pers_model = model.PersModel(nmod=len(MODS), nfeat=N_FEATS, nhid=N_FEATS // 2, dropout=DP).to(device)

    # initialize m2p2 models and hyper-parameters and optimizer
    m2p2_models = latent_models
    m2p2_models['align'] = align_model
    m2p2_models['pers'] = pers_model

    m2p2_params = get_hyper_params(m2p2_models)
    weight_mod = {mod: torch.tensor(1. / len(MODS), requires_grad=True) for mod in MODS}
    m2p2_params += list(weight_mod.values())

    m2p2_optim = optim.Adam(m2p2_params, lr=LR, weight_decay=W_DECAY)
    m2p2_scheduler = optim.lr_scheduler.StepLR(m2p2_optim, step_size=STEP_SIZE, gamma=SCHE_GAMMA)

    if VERBOSE:
        print('####### total m2p2 hyper-parameters ', count_hyper_params(m2p2_params))
        for k, v in m2p2_models.items():
            print(v)
            print(count_hyper_params(v.parameters()))

    # 3 - Initialize concat weights: w_a, w_v, w_l
    # weight_mod = {mod: 1. / len(MODS) for mod in MODS}

    # 4 - Train or Test
    if not TEST_MODE:
        min_loss_pers = 1e5
        #### Master Procedure Start ####
        # bar_N = tqdm(range(N_EPOCHS), desc='master procedure')
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            # #### Slave Procedure Start ####
            # # train ref model
            # bar_n = tqdm(range(n_EPOCHS), desc='slave procedure')
            # for slave_epoch in bar_n:
            #     train_loss_ref = train_ref(m2p2_models, ref_model, MODS, tra_loader, ref_optim, ref_scheduler)
            #
            # # eval ref model and update weight_mod
            # eval_loss_ref_mod = eval_ref(m2p2_models, ref_model, MODS, val_loader)
            # weight_mod = update_weight_mod(MODS, old_weight_mod=weight_mod, loss_ref_mod=eval_loss_ref_mod)
            # #### Slave Procedure End ####

            # train m2p2 model
            train_loss_align, train_loss_pers, train_acc = train_m2p2(m2p2_models, MODS, tra_loader, m2p2_optim, m2p2_scheduler, weight_mod)
            # eval and save m2p2 model
            eval_loss_align, eval_loss_pers, eval_acc = eval_m2p2(m2p2_models, MODS, val_loader, weight_mod)
            if eval_loss_pers < min_loss_pers:
                print(f'[SAVE MODEL] eval pers loss: {eval_loss_pers:.5f}\tmini pers loss: {min_loss_pers:.5f}')
                min_loss_pers = eval_loss_pers
                saveModel(FOLD, m2p2_models, weight_mod)

            # output loss information
            end_time = time.time()
            if VERBOSE:
                epoch_mins, epoch_secs = calc_epoch_time(start_time, end_time)
                print(f'Epoch: {epoch + 1:02}/{N_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain alignment loss:{train_loss_align:.5f}\tTrain persuasion loss:{train_loss_pers:.5f}\tTrain MAE Loss:{train_acc:.5f}')
                print(f'\tEval alignment loss:{eval_loss_align:.5f}\tEval persuasion loss:{eval_loss_pers:.5f}\tEval MAE Loss:{eval_acc:.5f}')
        #### Master Procedure End ####
    else:
        m2p2_models, weight_mod = loadModel(FOLD, m2p2_models)
        test_loss_align, test_loss_pers, test_acc = eval_m2p2(m2p2_models, MODS, tes_loader, weight_mod)
        print(f'Test alignment loss:{test_loss_align:.5f}\tTest persuasion loss:{test_loss_pers:.5f}\tTest MAE Loss:{test_acc:.5f}')
        print('MSE:', round(test_loss_pers, 3))
