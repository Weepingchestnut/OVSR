import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.common import generate_it, UPSCALE, PFRB


class UNIT(nn.Module):
    def __init__(self, kind='successor', basic_feature=64, num_frame=3, num_b=5, scale=4, act=nn.LeakyReLU(0.2, True)):
        super(UNIT, self).__init__()
        self.bf = basic_feature
        self.nf = num_frame
        self.num_b = num_b
        self.scale = scale
        self.act = act
        self.kind = kind
        if kind == 'precursor':
            self.conv_c = nn.Conv2d(3, self.bf, 3, 1, 3 // 2)
            self.conv_sup = nn.Conv2d(3 * (num_frame - 1), self.bf, 3, 1, 3 // 2)
        else:
            self.conv_c = nn.Sequential(*[nn.Conv2d((3 + self.bf), self.bf, 3, 1, 3 // 2) for i in range(num_frame)])
        self.blocks = nn.Sequential(*[PFRB(self.bf, 3, act) for i in range(num_b)])
        self.merge = nn.Conv2d(3 * self.bf, self.bf, 3, 1, 3 // 2)
        self.upscale = UPSCALE(self.bf, scale, act)
        print(kind, num_b)

    def forward(self, it, ht_past, ht_now=None, ht_future=None):
        B, C, T, H, W = it.shape    # torch.Size([16, 3, 3, 64, 64])

        if self.kind == 'precursor':
            it_c = it[:, :, T // 2]     # torch.Size([16, 3, 64, 64])   get reference frame
            index_sup = list(range(T))      # [0, 1, 2]
            index_sup.pop(T // 2)           # [0, 2]    去除参考帧的index，即得到近邻帧的index
            it_sup = it[:, :, index_sup]    # torch.Size([16, 3, 2, 64, 64])    get support frame
            it_sup = it_sup.view(B, C * (T - 1), H, W)      # torch.Size([16, 6, 64, 64])   channel * (3-1)
            hsup = self.act(self.conv_sup(it_sup))      # torch.Size([16, 56, 64, 64])  get support frame hidden states
            hc = self.act(self.conv_c(it_c))    # torch.Size([16, 56, 64, 64])  get reference frame hidden states
            inp = [hc, hsup, ht_past]       # 0: hc;    1: hsup;    2: ht_past
        else:
            ht = [ht_past, ht_now, ht_future]
            it_c = [torch.cat([it[:, :, i, :, :], ht[i]], 1) for i in range(3)]
            inp = [self.act(self.conv_c[i](it_c[i])) for i in range(3)]

        inp = self.blocks(inp)      # Residual Block    inp[0, 1, 2]: torch.Size([16, 56, 64, 64])

        ht = self.merge(torch.cat(inp, 1))      # channel dim concat    ht: torch.Size([16, 56, 64, 64])
        it_sr = self.upscale(ht)        # it_sr: torch.Size([16, 3, 256, 256])

        return it_sr, ht


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.bf = config.model.basic_filter
        self.num_pb = config.model.num_pb
        self.num_sb = config.model.num_sb
        self.scale = config.model.scale
        self.nf = config.model.num_frame
        self.kind = config.model.kind  # local or global
        self.act = nn.LeakyReLU(0.2, True)
        self.precursor = UNIT('precursor', self.bf, self.nf, self.num_pb, self.scale, self.act)
        self.successor = UNIT('successor', self.bf, self.nf, self.num_sb, self.scale, self.act)
        print(self.kind, '{}+{}'.format(self.num_pb, self.num_sb))

        params = list(self.parameters())
        pnum = 0
        for p in params:
            l = 1
            for j in p.shape:
                l *= j
            pnum += l
        print('Number of parameters {}'.format(pnum))

    def forward(self, x, start=0):
        B, C, T, H, W = x.shape
        # B, T, C, H, W = x.shape
        """
            print("x.shape =", x.shape)
            val: x.shape = torch.Size([1, 3, 9, 128, 240])
            train: x.shape = torch.Size([16, 3, 9, 64, 64])
        """
        start = max(0, start)
        end = T - start

        sr_all = []
        pre_sr_all = []
        pre_ht_all = []
        ht_past = torch.zeros((B, self.bf, H, W), dtype=torch.float, device=x.device)   # torch.Size([16, 56, 64, 64])

        # precursor
        for idx in range(T):
            t = idx if self.kind == 'local' else T - idx - 1
            insert_idx = T + 1 if self.kind == 'local' else 0

            it = generate_it(x, t, self.nf, T)      # torch.Size([16, 3, 3, 64, 64])
            it_sr_pre, ht_past = self.precursor(it, ht_past, None, None)        # it_sr_pre: torch.Size([16, 3, 256, 256]); ht_past: torch.Size([16, 56, 64, 64])
            pre_ht_all.insert(insert_idx, ht_past)
            pre_sr_all.insert(insert_idx, it_sr_pre)

        # successor
        ht_past = torch.zeros((B, self.bf, H, W), dtype=torch.float, device=x.device)
        for t in range(end):
            it = generate_it(x, t, self.nf, T)
            ht_future = pre_ht_all[t] if t == T - 1 else pre_ht_all[t + 1]
            it_sr, ht_past = self.successor(it, ht_past, pre_ht_all[t], ht_future)
            sr_all.append(it_sr + pre_sr_all[t])

        sr_all = torch.stack(sr_all, 2)[:, :, start:]
        """
            print("sr_all =", sr_all.size())
            val: sr_all = torch.Size([1, 3, 9, 512, 960])
            train: sr_all = torch.Size([16, 3, 7, 256, 256])
        """
        pre_sr_all = torch.stack(pre_sr_all, 2)[:, :, start:end]
        """
            print("pre_sr_all =", pre_sr_all.size())
            val: pre_sr_all = torch.Size([1, 3, 9, 512, 960])
            train: pre_sr_all = torch.Size([16, 3, 7, 256, 256])
        """

        return sr_all, pre_sr_all
