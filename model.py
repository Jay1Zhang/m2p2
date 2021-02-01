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

from utils import *


# 0 - Init Module
# The model generate primary input embedding of one modality
class InputEmb(nn.Module):
    def __init__(self, ninp, feat_dim=16, dropout=0.1):
        super(InputEmb, self).__init__()
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
        # Input:    src(S, N, E:16),  seq_msk(N, S)
        # Output:   out(T, N, E:16)
        src *= math.sqrt(self.ninp)
        # positional encoder
        src = self.pos_encoder(src)
        # transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=seq_msk)
        return output


# positional encoding
# 无可学习参数的PositionEncoding层
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
class LatentModel(nn.Module):
    def __init__(self, mod, nfeat=16, nhid=8, dropout=0.1):
        super(LatentModel, self).__init__()

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
        # Input:    src(N, S, E),  seq_msk(N, S)
        #           where N: batch size, S: sequence length, E: {a:73, v:512, l:200}
        # Output:   out(T, N, 16)
        N, S = src.size()[0], src.size()[1]
        feats = torch.stack([self.feat_exa(src[i]) for i in range(N)], dim=0).transpose(0, 1)
        # feats: (S,N,16)
        seq = self.transformer_emb(feats, seq_msk)  # (S, N, 16)
        seq = F.relu(seq)
        # max_pool
        msked_tmp = seq * (~seq_msk.unsqueeze(-1).transpose(0, 1)).float()  # (220, N, 16)
        out = torch.max(msked_tmp, dim=0)[0]    # (N, 16)
        return out

############################## question end ##################################


# 1 - Alignment Module
# Get alignment embeddings $H^s_m$ with $H^latent_m$
class AlignModel(nn.Module):
    def __init__(self, ninp, nout, dropout=0.1):
        super(AlignModel, self).__init__()
        self.fc1 = nn.Linear(ninp, nout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent_emb_mod):
        # get $H^s_m$ with $H^latent_m$
        s_emb_mod = {}      # H^s_m
        for mod, latent_emb in latent_emb_mod.items():
            x = self.fc1(latent_emb)
            x = self.dropout(x)
            x = F.relu(x)
            s_emb_mod[mod] = x

        return s_emb_mod    # H^s_m


# 2 - Heterogeneity Module
# Note: 为了保持Align与Het的统一，把三个ref model也拼在一起训练，相当于也用虚线圈起来了
# the uni-modal reference models, used to calculate $L^ref_m$ and update weights: $w_m$
class HetModel(nn.Module):
    def __init__(self, nfeat=16, nhid=8, dropout=0.1):
        super(HetModel, self).__init__()

        ninp = nfeat
        nout = 1
        self.fc1 = nn.Linear(ninp, 2 * nhid)
        self.fc2 = nn.Linear(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)
        self.sigm = nn.Sigmoid()

    def forward(self, latent_emb_mod):
        # get $Y^ref_m$ with $H^latent_m$
        ref_emb_mod = {}
        for mod, latent_emb in latent_emb_mod.items():
            x = self.fc1(latent_emb)
            x = F.relu(self.dropout(x))
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            x = self.sigm(x)
            ref_emb_mod[mod] = x

        return ref_emb_mod


# 3 - Persuasiveness Module
class PersModel(nn.Module):
    def __init__(self, nmod=3, nfeat=16, nhid=8, dropout=0.1):
        super(PersModel, self).__init__()

        # input: heterogeneity emb (nmod * nfeat), alignment emb (nfeat), debate meta-data (1)
        ninp = (nmod+1) * nfeat + 2
        nout = 1
        self.fc1 = nn.Linear(ninp, 2 * nhid)
        self.fc2 = nn.Linear(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)
        self.sigm = nn.Sigmoid()

    def forward(self, align_emb, het_emb, meta_emb):
        # x = torch.cat([align_emb, het_emb, meta_emb], dim=1)
        x = torch.cat([het_emb, align_emb, meta_emb], dim=1)
        x = self.fc1(x)
        x = F.relu(self.dropout(x))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return self.sigm(x)

