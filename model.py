#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import math
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models
from torch.nn.modules.container import ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# 0 - Init Module
# The model generate primary input embedding of one modality
class InputEmb(nn.Module):
    def __init__(self, ninp, feat_dim=16, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(ninp, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x = self.fc(src)
        x = self.bn(x)
        return F.relu(self.dropout(x))


############################## question here ##################################
# positional encoder + transformer encoder
class TransformerEmb(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers=1, dropout=0.1):
        super(TransformerEmb, self).__init__()
        # self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # 1-layer transformer encoder
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, seq_msk):
        src *= math.sqrt(self.ninp)
        # positional encoder
        src = self.pos_encoder(src)
        # transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=seq_msk)
        return output


# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# The model generate the compact latent embedding of one modality
class LatentEmb(nn.Module):
    def __init__(self, mod, nfeat=16, nhid=8, dropout=0.1):
        super(LatentEmb, self).__init__()

        # question: it means?
        # answer:
        nhead = 4       # head attention ???
        nlayers = 1     # transformer encoder layers ???

        if mod == 'a':
            self.feat_exa = InputEmb(73, nfeat, dropout)
        elif mod == 'v':
            self.feat_exa = InputEmb(512, nfeat, dropout)
        else:
            self.feat_exa = InputEmb(200, nfeat, dropout)

        self.transformer_emb = TransformerEmb(nfeat, nhead, nhid, nlayers, dropout)

    def forward(self, src, seq_msk):
        # N: batch size, S: sequence length
        N, S = src.size()[0], src.size()[1]
        feats = torch.stack(
            [self.feat_exa(src[i]) for i in range(N)],
            dim=0).transpose(0, 1)
        # feats: (S,N,D)
        seq = self.transformer_emb(feats, seq_msk)
        seq = F.relu(seq)
        # max_pool
        msked_tmp = seq * (~seq_msk.unsqueeze(-1).transpose(0, 1)).float()
        out = torch.max(msked_tmp, dim=0)[0]
        return out

############################## question end ##################################


# 1 - Alignment Module
# Get alignment embeddings $H^align$ and $H^s_m$ with $H^latent_m$
class AlignEmb(nn.Module):
    def __init__(self, ninp, nout, dropout=0.1):
        super(AlignEmb, self).__init__()
        self.fc1 = nn.Linear(ninp, nout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent_emb_mod):
        # get $H^s_m$ with $H^latent_m$
        s_emb_mod = {}      # H^s_m
        for mod, latent_emb in latent_emb_mod.items():
            s_emb_mod[mod] = F.relu(self.dropout(self.fc1(latent_emb)))

        # get $H^align$ with $H^s_m$
        align_cat = torch.cat([emb.unsqueeze(dim=0) for emb in s_emb_mod.values()], dim=0)
        align_emb = torch.mean(align_cat, dim=0)    # H^align

        # return: H^align, H^s_m
        return align_emb, s_emb_mod


# 2 - Heterogeneity Module
# Note: 注意Align与Het的区别，Align可以拼在一起，而Het中的ref是三个分开训的
# the uni-modal reference models, used to calculate $L^ref_m$ and update weights: $w_m$
class RefModel(nn.Module):
    def __init__(self, nfeat=16, nhid=8, dropout=0.1):
        super(RefModel, self).__init__()

        ninp = nfeat
        nout = 1
        self.fc1 = nn.Linear(ninp, 2 * nhid)
        self.fc2 = nn.Linear(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)
        self.sigm = nn.Sigmoid()

    def forward(self, latent_emb):
        # get $Y^ref_m$ with $H^latent_m$
        # question: why dim=1?
        # answer: 我认为完全无必要，因为align是cat在一起训练的，而ref是分开训练的，这里只会传进来对应mod的emb供训练
        # x = torch.cat([v for k, v in latent_emb_mod.items()], dim=1)

        x = self.fc1(latent_emb)
        x = F.relu(self.dropout(x))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return self.sigm(x)

# 3 - Persuasiveness Module

