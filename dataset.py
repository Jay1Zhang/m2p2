#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import utils

# get frame number: filename[st:ed] from the filename string
def get_frame_no(filename, form='npy'):
    # return filename[filename.rfind('/') + 1: filename.rfind(f'.{form}')]  # Linux
    return filename[filename.rfind('\\') + 1 : filename.rfind(f'.{form}')]  # Win10

# qps dataset class
class qpsDataset(Dataset):
    def __init__(self, mods, meta, segs):
        self.mods = mods
        self.meta = meta
        self.segs = segs

        self.len = len(self.segs)

        self.feat_src = './qps_dataset/'
        self.loadFeats = {'a': self.load_audio_feat, 'v': self.load_video_feat, 'l': self.load_lang_feat}

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ed_vote, change, uid, dur_sec = self.meta.loc[self.segs[index][0], ['ed_vote', 'change', 'uid', 'dur_sec']]
        sample = {'ed_vote': ed_vote, 'change': change, 'uid': uid, 'dur': dur_sec}

        for mod in self.mods:
            data_list, msk_list = [], []
            used_segs = self.segs[index]
            for seg_id in used_segs:
                x, y = self.loadFeats[mod](f'{seg_id:04d}')
                data_list.append(x)
                msk_list.append(y)
            data = torch.cat(data_list, 0)
            msk = torch.cat(msk_list, 0)

            # questions here: why padding?

        return sample


    # load acoustic features from covarep_norm.npy
    def load_audio_feat(self, seg):
        feat = np.load(f'{self.feat_src}/{seg}/covarep_norm.npy')
        # questions here: 为什么msk都给了false？这不是元数据吗？
        return torch.from_numpy(feat), torch.zeros([feat.shape[0]], dtype=torch.bool)

    # load video features from frames in vgg_1fc/
    def load_video_feat(self, seg):
        form = 'npy'
        imgs, msk = [], []
        filenames = np.sort(np.array(glob.glob(f'{self.feat_src}/{seg}/vgg_1fc/*{form}')))

        min_frame = int(get_frame_no(filenames[0], form))
        max_frame = int(get_frame_no(filenames[-1], form))

        # questions here
        for frame in range(min_frame, max_frame+1):
            filename = f'{self.feat_src}/{seg}/vgg_1fc/{frame:05}.{form}'
            pass

    # load language features from tencent_emb.npy
    def load_lang_feat(self, seg):
        feat = np.load(f'{self.feat_src}/{seg}/tencent_emb.npy').astype(np.float32)
        # questions here: 为什么msk都给了false？这不是元数据吗？
        return torch.from_numpy(feat), torch.zeros([feat.shape[0]], dtype=torch.bool)

