import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class cha_loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(cha_loss, self).__init__()
        self.eps = eps
        return

    def forward(self, inp, target):
        diff = torch.abs(inp - target) ** 2 + self.eps ** 2
        out = torch.sqrt(diff)
        loss = torch.mean(out)

        return loss


def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f - 1).tolist()
    it = x[:, :, index]

    return it


class UPSCALE(nn.Module):
    def __init__(self, basic_feature=64, scale=4, act=nn.LeakyReLU(0.2, True)):
        super(UPSCALE, self).__init__()
        body = []
        body.append(nn.Conv2d(basic_feature, 48, 3, 1, 3 // 2))
        body.append(act)
        body.append(nn.PixelShuffle(2))
        body.append(nn.Conv2d(12, 12, 3, 1, 3 // 2))
        body.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class PFRB(nn.Module):
    """
    Progressive Fusion Residual Block
    Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations, ICCV 2019
    """

    def __init__(self, basic_feature=64, num_channel=3, act=torch.nn.LeakyReLU(0.2, True)):
        super(PFRB, self).__init__()
        self.bf = basic_feature     # bf = 56
        self.nc = num_channel       # nc = 3
        self.act = act
        self.conv0 = nn.Sequential(*[nn.Conv2d(self.bf, self.bf, 3, 1, 3 // 2) for _ in range(num_channel)])
        self.conv1 = nn.Conv2d(self.bf * num_channel, self.bf, 1, 1, 1 // 2)
        self.conv2 = nn.Sequential(*[nn.Conv2d(self.bf * 2, self.bf, 3, 1, 3 // 2) for _ in range(num_channel)])

    def forward(self, x):
        x1 = [self.act(self.conv0[i](x[i])) for i in range(self.nc)]    # x[0]: hc; x[1]: hsup; x[2]: ht_past
        merge = torch.cat(x1, 1)    # torch.Size([16, 168, 64, 64])
        base = self.act(self.conv1(merge))      # 1x1Conv: torch.Size([16, 56, 64, 64])
        x2 = [torch.cat([base, i], 1) for i in x1]
        x2 = [self.act(self.conv2[i](x2[i])) for i in range(self.nc)]

        return [torch.add(x[i], x2[i]) for i in range(self.nc)]


class Pos2Weight(nn.Module):
    def __init__(self, inC, kernel_size=3, outC=3):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output


class MetaUpscale(nn.Module):
    def __init__(self, scale: float, planes: int = 64, act_mode: str = 'relu', use_affine: bool = True):
        super(MetaUpscale, self).__init__()
        self.scale = scale
        # self.scale_int = math.ceil(self.scale)
        self.P2W = Pos2Weight(inC=planes)

    def forward(self, x, res, pos_mat):
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))  # (outH*outW, outC*inC*kernel_size*kernel_size)
        up_x = self.repeat_x(res)   # the output is (N*r*r,inC,inH,inW)

        # N*r^2 x [inC * kH * kW] x [inH * inW]
        cols = F.unfold(up_x, 3, padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2), scale_int, x.size(3), scale_int, -1, 3).permute(1, 3,
                                                                                                                 0, 2,
                                                                                                                 4,
                                                                                                                 5).contiguous()

        local_weight = local_weight.contiguous().view(scale_int ** 2, x.size(2) * x.size(3), -1, 3)

        out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        out = out.contiguous().view(x.size(0), scale_int, scale_int, 3, x.size(2), x.size(3)).permute(0, 3, 4, 1, 5, 2)
        out = out.contiguous().view(x.size(0), 3, scale_int * x.size(2), scale_int * x.size(3))

        return out

    def repeat_x(self, x):
        scale_int = math.ceil(self.scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return x.contigupus().view(-1, C, H, W)
