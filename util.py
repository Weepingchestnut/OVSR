import random
import sys
import os
import time
import numpy as np
from os.path import join, exists
import glob

from torch import Tensor, cuda, squeeze, chunk, cat, stack
from tqdm import trange, tqdm
import cv2
import math
import scipy
import torch
from torch.nn import functional as F
import json

from core import imresize
from mytest.numpy_test import numpy_test


def automkdir(path):
    if not exists(path):
        os.makedirs(path)


def automkdirs(path):
    [automkdir(p) for p in path]


def compute_psnr_torch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1. / mse)


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log(1. / mse) / np.log(10)


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()  # torch.Size([1, 3, 9, 512, 960])
    x = x.view(-1, 1, H, W)  # torch.Size([27, 1, 512, 960])
    filter_height, filter_width = 13, 13
    pad_w, pad_h = (filter_width - 1) // 2, (filter_height - 1) // 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        if H % 3 != 0:
            r_h = 3 - (H % 3)
        if W % 3 != 0:
            r_w = 3 - (W % 3)
    x = F.pad(x, (pad_w, pad_w + r_w, pad_h, pad_h + r_h), 'reflect')

    gaussian_filter = torch.from_numpy(gkern(filter_height, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def makelr_fromhr_cuda(hr, scale=4, device=None, data_kind='single', down_sample='bi'):
    if data_kind == 'double' or isinstance(hr, (tuple, list)):
        return [i.to(device) for i in hr]
    else:
        hr = hr.to(device)
        if down_sample == 'bi':
            lr = BI_downsample(hr, scale)
        else:
            lr = DUF_downsample(hr, scale)
        return lr, hr


def save_all_srframe(cleans, config):
    sr_all = [(np.round(np.clip(cleans[0].cpu().numpy()[0, i] * 255, 0, 255))).astype(np.uint8)
              for i in range(config.train.num_frame)]
    # print("sr_all[0].shape: ", sr_all[0].shape)
    for num_frames in range(config.train.num_frame):
        cv2_imsave(join(config.path.eval_result, '{:0>4}.png'.format(num_frames)), sr_all[num_frames])


def evaluation(model, eval_data, config):
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    psnr_all = []
    scale = config.model.val_scale
    epoch = config.train.epoch
    device = config.device
    test_runtime = []
    in_h = 128
    in_w = 240
    bd = 2

    for iter_eval, (img_hq) in enumerate(tqdm(eval_data)):
        img_hq = img_hq[:, :, :, int(bd * scale + 0.5): int((bd + in_h) * scale + 0.5), int(bd * scale + 0.5): int((bd + in_w) * scale + 0.5)]
        # img_hq: torch.Size([1, 3, 9, bd:2 * scale:4 : (bd:2 + in_h:128)*scale:4 = 512, 960])
        img_lq, img_hq = makelr_fromhr_cuda(img_hq, scale, device, config.data_kind, config.data_downsample)
        # img_lq = img_lq[:, :, :, :in_h, :in_w]
        # img_hq = img_hq[:, :, :, :in_h*scale, :in_w*scale]

        B, C, T, H, W = img_lq.shape  # img_lq: torch.Size([1, 3, 9, 128, 240])
        # B, T, C, H, W = img_lq.shape    # torch.Size([1, 9, 3, 130, 244])   scale=3.5

        start.record()
        with torch.no_grad():
            img_clean = model(img_lq)
            # print("img_clean = ", img_clean.size())
        if type(scale) is not int:
            img_clean_resize = [BI_resize(_, (img_hq.size(3), img_hq.size(4)))
                                for _ in img_clean]
        else:
            img_clean_resize = img_clean
        end.record()
        torch.cuda.synchronize()  # test model inference time more correct
        test_runtime.append(start.elapsed_time(end) / T)

        # cleans = [_.permute(0, 2, 3, 4, 1) for _ in img_clean]
        cleans = [_.permute(0, 2, 3, 4, 1) for _ in img_clean_resize]
        # print("len(cleans) = ", len(cleans))      # len(cleans) = 2
        hr = img_hq.permute(0, 2, 3, 4, 1)
        # print("hr = ", hr.size())     # hr = torch.Size([1, 9, 512, 960, 3])

        psnr_cleans, psnr_hr = cleans, hr
        psnrs = [compute_psnr_torch(_, psnr_hr).cpu().numpy() for _ in psnr_cleans]
        # numpy_test(cleans[0].cpu().numpy())      # test
        if config.eval.save_all_sr == 1:  # save all SR sequences
            save_all_srframe(cleans, config)
        else:
            clean = (np.round(np.clip(cleans[0].cpu().numpy()[0, T // 2] * 255, 0, 255))).astype(np.uint8)
            cv2_imsave(join(config.path.eval_result, '{:0>4}.png'.format(iter_eval)), clean)
        # clean = (np.round(np.clip(cleans[0].cpu().numpy()[0, T // 2] * 255, 0, 255))).astype(np.uint8)
        # cv2_imsave(join(config.path.eval_result, '{:0>4}.png'.format(iter_eval)), clean)
        psnr_all.append(psnrs)

    psnrs = np.array(psnr_all)
    psnr_avg = np.mean(psnrs, 0, keepdims=False)

    with open(config.path.eval_file, 'a+') as f:
        eval_dict = {'Epoch': epoch, 'PSNR': psnr_avg.tolist()}
        eval_json = json.dumps(eval_dict)
        f.write(eval_json)
        f.write('\n')
    print(eval_json)
    ave_runtime = sum(test_runtime) / len(test_runtime)
    print(f'average time cost {ave_runtime} ms')

    model.train()

    return psnr_avg


def test_video(model, path, savepath, config):
    model.eval()
    automkdir(savepath)
    scale = config.model.scale
    device = config.device
    # print(savepath)
    prefix = os.path.split(path)[-1]
    inp_type = 'truth' if config.data_kind == 'single' else f'input{config.model.scale}'
    imgs = sorted(glob.glob(join(path, inp_type, '*.png')))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    test_runtime = []

    if inp_type == 'truth':
        img_hq = [cv2_imread(i) for i in imgs]
        img_hq = torch.from_numpy(np.array(img_hq) / 255.).float().permute(3, 0, 1, 2).contiguous()
        img_hq = img_hq.to(device)
        img_lq = DUF_downsample(img_hq.unsqueeze(0), scale)
    else:
        img_lq = [cv2_imread(i) for i in imgs]
        img_lq = torch.from_numpy(np.array(img_lq)).float().permute(3, 0, 1, 2).contiguous() / 255.
        img_lq = img_lq.to(device).unsqueeze(0)
    B, C, T, H, W = img_lq.shape

    files_info = [os.path.split(_)[-1] for _ in imgs]

    start.record()
    with torch.no_grad():
        img_clean = model(img_lq)
    end.record()
    torch.cuda.synchronize()
    test_runtime.append(start.elapsed_time(end))  # milliseconds

    if isinstance(img_clean, tuple):
        img_clean = img_clean[0]

    sr = img_clean[0].permute(1, 2, 3, 0)
    sr = sr.cpu().numpy()
    sr = (np.round(np.clip(sr * 255, 0, 255))).astype(np.uint8)
    [cv2_imsave(join(savepath, files_info[i]), sr[i]) for i in range(T)]
    print('Cost {} ms in average.\n'.format(np.mean(test_runtime) / T))

    return


def save_checkpoint_psnr(model, epoch, model_folder, psnr):
    # --------------------------------------------------------------
    # checkpoint file's name example:
    #       print('{:0>4}_psnr-{:.4f}.pth'.format(1, 31.11119999))
    # --------------------------------------------------------------
    # model_out_path = os.path.join(model_folder, '{:0>4}.pth'.format(epoch))
    model_out_path = os.path.join(model_folder, '{:0>4}_psnr-{:.4f}.pth'.format(epoch, psnr))
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    return


def save_checkpoint(model, epoch, model_folder):
    model_out_path = os.path.join(model_folder, '{:0>4}.pth'.format(epoch))
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    return


def load_checkpoint(network=None, resume='', path='', weights_init=None, rank=0):
    try:
        num_resume = int(resume[resume.rfind('/') + 1:resume.rfind('.')])
    except Exception as e:
        num_resume = 0
    finally:
        if num_resume < 0:
            checkpointfile = sorted(glob.glob(join(path, '*')))
            if len(checkpointfile) == 0:
                resume = 'nofile'
            else:
                resume = checkpointfile[-1]
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage.cuda(rank))
            start_epoch = checkpoint["epoch"]
            network.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            if weights_init is not None:
                network.apply(weights_init)
                # print("weights_init")
            start_epoch = 0

    return start_epoch


def adjust_learning_rate(init_lr, final_lr, epoch, epoch_decay, iteration, iter_per_epoch, optimizer, ifprint=False):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = (init_lr - final_lr) * max((1 - (epoch + iteration / iter_per_epoch) / epoch_decay), 0) + final_lr
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

    if ifprint:
        print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    return lr


def cv2_imsave(img_path, img):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def cv2_imread(img_path):
    img = cv2.imread(img_path)  # img.shape = {tuple}(792, 1920, 3)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    return img


class DICT2OBJ(object):
    def __init__(self, obj, v=None):
        # if not isinstance(obj, dict):
        #     setattr(self, obj, v)
        #     return 
        for k, v in obj.items():
            if isinstance(v, dict):
                # print('dict', k, v)
                setattr(self, k, DICT2OBJ(v))
            else:
                # print('no dict', k, v)
                setattr(self, k, v)


def BI_resize(x: Tensor, size):
    if cuda.is_available() and (not x.is_cuda):
        x = x.cuda()

    # B, C, T, H, W --> B, T, C, H, W
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    B, T, C, H, W = x.size()

    x = x.view(-1, C, H, W)
    x = imresize(x, sizes=size)

    x = x.view(B, T, C, x.size(2), x.size(3))
    x = x.permute(0, 2, 1, 3, 4).contiguous()

    return x


def BI_downsample(x: Tensor, scale: float = 4.0):
    """Bicubic Downsamping with used in bicubic_pytorch code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (float): downsampling factor: (1.0, 4.0].

    Returns:
        x (Tensor, [B, T, C, H/scale, W/scale]): frames after downsampled.
    """
    if cuda.is_available() and (not x.is_cuda):
        x = x.cuda()

    # print("input x.shape =", x.size())
    # B, C, T, H, W --> B, T, C, H, W
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    B, T, C, H, W = x.size()
    # print("after permute x.shape =", x.size())
    # print("T =", T)

    scale_1 = round(1 / scale, 3)
    # print("scale_1:", scale_1)
    down_H = int(H / scale + 0.5)
    down_W = int(W / scale + 0.5)

    x = x.view(-1, C, H, W)
    # print("after view:", x.size())
    x = imresize(x, sizes=(down_H, down_W))
    # print("after resize:", x.size())

    x = x.view(B, T, C, x.size(2), x.size(3))
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    # print("after view and permute:", x.size())

    return x


if __name__ == '__main__':
    # print('{:0>4}_psnr-{:.4f}.pth'.format(1, 31.11119999))
    # test_video()
    test_x = torch.arange(16 * 3 * 9 * 24 * 24).float().view(16, 3, 9, 24, 24)
    scale = [s / 10 for s in list(range(11, 41, 1))]
    for i in range(100):
        idx_scale = random.randrange(0, len(scale))
        print(scale[idx_scale])
        resize_x = BI_downsample(test_x, scale[idx_scale])
        # print("resize_x:", resize_x.size())
        print("~" * 100)

    # b, t, c, h, w
    # channel = 3
    # T = 5
    # test_tc = torch.arange(2*T*channel*4*4).float().view(2, T, channel, 4, 4)
    # test_ct = torch.arange(2*channel*T*4*4).float().view(2, channel, T, 4, 4)
    #
    # resize_tc = BI_downsample(test_tc, 2.0)
    # print("="*100)
    # resize_ct = BI_downsample(test_ct, 2.0)
    #
    # print("resize_tc = \n", resize_tc)
    # print("resize_ct = \n", resize_ct)
