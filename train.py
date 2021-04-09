#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch

from utils import *


def gen_align_emb(s_emb_mod):
    # generate H^align with H^s_m
    align_cat = torch.cat([emb.unsqueeze(dim=0) for emb in s_emb_mod.values()], dim=0)
    align_emb = torch.mean(align_cat, dim=0)  # H^align
    return align_emb


def gen_het_emb(latent_emb_mod, weight_mod, MODS):
    # generate H^het with H^latent_m, w_m
    het_emb_mod = {}
    for mod in MODS:
        het_emb_mod[mod] = weight_mod[mod] * latent_emb_mod[mod]
    # het_emb_mod = [torch.tensor(weight_mod[mod]) * latent_emb_mod[mod] for mod in MODS]
    het_emb = torch.cat([v for v in het_emb_mod.values()], dim=1)
    return het_emb, het_emb_mod


def gen_meta_emb(sample):
    st_vote = (sample['ed_vote'] - sample['change']).float().unsqueeze(1).to(device)
    dur_time = (sample['dur'] / MAX_DUR).float().unsqueeze(1).to(device)
    meta_emb = torch.cat([st_vote, dur_time], dim=1)
    return meta_emb


def fit_m2p2(m2p2_models, MODS, sample_batched, weight_mod):
    latent_emb_mod = {}
    for mod in MODS:
        latent_emb_mod[mod] = m2p2_models[mod](sample_batched[f'{mod}_data'].to(device),
                                               sample_batched[f'{mod}_msk'].to(device))

    # s_emb_mod = m2p2_models['align'](latent_emb_mod)
    # align_emb = gen_align_emb(s_emb_mod)
    het_emb, het_emb_mod = gen_het_emb(latent_emb_mod, weight_mod, MODS)
    meta_emb = gen_meta_emb(sample_batched)

    y_pred = m2p2_models['pers'](None, het_emb, meta_emb)
    y_true = sample_batched['ed_vote'].float().to(device)

    # calc loss
    #loss_align = calcAlignLoss(s_emb_mod, MODS)
    loss_pers = calcPersLoss(y_pred, y_true)
    loss = loss_pers
    #loss = loss_pers + GAMMA * loss_align  # final loss, used to backward
    acc = calcAccuracy(y_pred, y_true)

    return None, loss_pers, loss, acc


def train_m2p2(m2p2_models, MODS, iterator, optimizer, scheduler, weight_mod):
    setModelMode(m2p2_models, is_train_mode=True)
    total_loss_align, total_loss_pers = 0, 0
    total_acc = 0

    for i_batch, sample_batched in enumerate(iterator):
        optimizer.zero_grad()
        # forward
        loss_align, loss_pers, loss, acc = fit_m2p2(m2p2_models, MODS, sample_batched, weight_mod)
        #total_loss_align += loss_align.item()
        total_loss_pers += loss_pers.item()
        total_acc += acc.item()

        # backward
        loss.backward()
        optimizer.step()

    scheduler.step()
    return total_loss_align / (i_batch+1), total_loss_pers / (i_batch+1), total_acc / (i_batch + 1)    # mean


def eval_m2p2(m2p2_models, MODS, iterator, weight_mod):
    setModelMode(m2p2_models, is_train_mode=False)
    total_loss_align, total_loss_pers = 0, 0
    total_acc = 0

    for i_batch, sample_batched in enumerate(iterator):
        # forward
        with torch.no_grad():
            loss_align, loss_pers, loss, acc = fit_m2p2(m2p2_models, MODS, sample_batched, weight_mod)
            #total_loss_align += loss_align.item()
            total_loss_pers += loss_pers.item()
            total_acc += acc.item()

    return total_loss_align / (i_batch+1), total_loss_pers / (i_batch+1), total_acc / (i_batch + 1)    # mean


def fit_ref(m2p2_models, ref_model, MODS, sample_batched):
    latent_emb_mod = {}
    for mod in MODS:
        latent_emb_mod[mod] = m2p2_models[mod](sample_batched[f'{mod}_data'].to(device),
                                               sample_batched[f'{mod}_msk'].to(device))

    ref_emb_mod = ref_model(latent_emb_mod)
    y_true = sample_batched['ed_vote'].float().to(device)

    # calc loss
    loss_ref_mod = {}
    for mod in MODS:
        y_pred = ref_emb_mod[mod]
        loss_ref_mod[mod] = calcPersLoss(y_pred, y_true)

    loss_ref = sum(list(loss_ref_mod.values()))
    return loss_ref, loss_ref_mod


def train_ref(m2p2_models, ref_model, MODS, iterator, optimizer, scheduler):
    setModelMode({'ref': ref_model}, is_train_mode=True)
    total_loss_ref = 0

    for i_batch, sample_batched in enumerate(iterator):
        optimizer.zero_grad()
        # forward
        loss_ref, loss_ref_mod = fit_ref(m2p2_models, ref_model, MODS, sample_batched)
        total_loss_ref += loss_ref.item()

        # backward
        loss_ref.backward()
        optimizer.step()

    scheduler.step()
    return total_loss_ref / (i_batch+1)


def eval_ref(m2p2_models, ref_model, MODS, iterator):
    setModelMode({'ref': ref_model}, is_train_mode=False)
    total_loss_ref_mod = {mod: 0 for mod in MODS}

    for i_batch, sample_batched in enumerate(iterator):
        # forward
        with torch.no_grad():
            loss_ref, loss_ref_mod = fit_ref(m2p2_models, ref_model, MODS, sample_batched)
            for mod in MODS:
                total_loss_ref_mod[mod] += loss_ref_mod[mod].item()

    for mod in MODS:
        loss_ref_mod[mod] = total_loss_ref_mod[mod] / (i_batch+1)

    return loss_ref_mod

