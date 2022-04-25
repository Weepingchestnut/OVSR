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


def input_matrix_wpn(inH, inW, scale, add_scale=True):
    """
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    """
    outH, outW = int(scale * inH), int(scale * inW)

    # mask records which pixel is invalid, 1 valid or o invalid,
    # h_offset and w_offset calculate the offset to generate the input matrix
    # （mask 记录哪些像素无效，1有效 or 0无效，h_offset 和 w_offset 计算偏移量以生成输入矩阵）
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH, scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int)
    if add_scale:
        scale_mat = torch.zeros(1, 1)
        scale_mat[0, 0] = 1.0 / scale
        # res_scale = scale_int - scale
        # scale_mat[0,scale_int-1]=1-res_scale
        # scale_mat[0,scale_int-2]= res_scale
        scale_mat = torch.cat([scale_mat] * (inH * inW * (scale_int ** 2)), 0)  # (inH*inW*scale_int**2, 4)

    # projection coordinate and calculate the offset
    # （投影坐标和计算偏移量）
    h_project_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale)
    int_h_project_coord = torch.floor(h_project_coord)

    offset_h_coord = h_project_coord - int_h_project_coord
    int_h_project_coord = int_h_project_coord.int()

    w_project_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale)
    int_w_project_coord = torch.floor(w_project_coord)

    offset_w_coord = w_project_coord - int_w_project_coord
    int_w_project_coord = int_w_project_coord.int()

    # flag for number for current coordinate LR image
    # （标记当前LR图像坐标的编号）
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag, 0] = 1
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1

    flag = 0
    number = 0
    for i in range(outW):
        if int_w_project_coord[i] == number:
            w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], flag] = 1
            flag += 1
        else:
            w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0] = 1
            number += 1
            flag = 1

    # the size is scale_int * inH * (scal_int * inW)
    h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
    #
    mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
    mask_mat = mask_mat.eq(2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    if add_scale:
        pos_mat = torch.cat((scale_mat.view(1, -1, 1), pos_mat), 2)

    return pos_mat, mask_mat  # outH*outW*2 outH=scale_int*inH , outW = scale_int *inW
