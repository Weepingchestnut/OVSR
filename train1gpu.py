import argparse
import importlib
import math
import random
import sys
import time
from os.path import join

import torch.cuda
import yaml
from ptflops import get_model_complexity_info
from torch.cuda.amp import GradScaler, autocast

import dataloader
from models.common import weights_init, cha_loss
from util import DICT2OBJ, automkdir, load_checkpoint, adjust_learning_rate, evaluation, makelr_fromhr_cuda, \
    save_checkpoint


def train(gpu_id, config):
    torch.cuda.set_device(gpu_id)
    config.device = device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    print("Random Seed: ", config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)

    model = config.network
    criterion = getattr(sys.modules[__name__], config.train.loss)()
    model = model.to(device)
    criterion = criterion.to(device)

    epoch = load_checkpoint(model, config.path.resume, config.path.checkpoint, weights_init=weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.init_lr, weight_decay=0)

    config.train.epoch = epoch
    iter_per_epoch = config.train.iter_per_epoch
    epoch_decay = config.train.epoch_decay
    step = 0
    scaler = GradScaler()

    train_batch_size = config.train.batch_size
    train_dataset = dataloader.loader(config.path.train, data_kind=config.data_kind, mode='train',
                                      scale=config.model.scale,
                                      crop_size=config.train.in_size, num_frame=config.train.num_frame)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                               num_workers=config.train.num_workers,
                                               pin_memory=False, drop_last=True)

    eval_dataset = dataloader.loader(config.path.eval, data_kind=config.data_kind, mode='eval',
                                     scale=config.model.scale,
                                     num_frame=config.train.num_frame)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.eval.batch_size, shuffle=False,
                                              num_workers=config.eval.num_workers,
                                              pin_memory=False)
    loss_frame_seq = list(range(config.train.sub_frame, config.train.num_frame - config.train.sub_frame))
    alpha = config.train.alpha

    while epoch < config.train.num_epochs:
        if step == 0:
            adjust_learning_rate(config.train.init_lr, config.train.final_lr, epoch, epoch_decay, step % iter_per_epoch,
                                 iter_per_epoch, optimizer, True)
            if gpu_id == 0:
                max_psnr = evaluation(model, eval_loader, config).tolist()[0]
                # print("0 max_psnr = ", max_psnr)
            time_start = time.time()

        for iteration, (img_hq) in enumerate(train_loader):
            adjust_learning_rate(config.train.init_lr, config.train.final_lr, epoch, epoch_decay, step % iter_per_epoch,
                                 iter_per_epoch, optimizer, False)
            optimizer.zero_grad()

            img_lq, img_hq = makelr_fromhr_cuda(img_hq, config.model.scale, device, config.data_kind)

            with autocast():  # Automatic Mixed Precision Training
                it_all, pre_it_all = model(img_lq, config.train.sub_frame)
                loss = criterion(it_all, img_hq[:, :, loss_frame_seq]) + alpha * criterion(pre_it_all,
                                                                                           img_hq[:, :, loss_frame_seq])

            loss_v = loss.detach()
            if (loss_v > 5 or loss_v < 0 or math.isnan(loss_v)) and epoch > 0:
                print(f'epoch {epoch}, skip iteration {iteration}, loss {loss_v}')
                raise RuntimeWarning(f'epoch {epoch}, iteration {iteration}, loss {loss_v}')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            step += 1

            # ----------------------------
            # print loss per display_iter
            # ----------------------------
            if (step % config.train.display_iter) == 0 and gpu_id == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      "Epoch[{}/{}]({}/{}): Loss: {:.8f}".format(epoch, config.train.num_epochs,
                                                                 step % iter_per_epoch, iter_per_epoch, loss_v))
                sys.stdout.flush()

            if step % iter_per_epoch == 0:

                # dist.barrier()
                # if gpu_id == 0:
                time_cost = time.time() - time_start
                print(f'spent {time_cost} s')
                # ------------------------------------------------
                # evaluation + compute psnr
                psnr_avg = evaluation(model, eval_loader, config)
                psnr_avg = psnr_avg.tolist()[0]
                # print("psnr_avg = ", psnr_avg)
                # ------------------------------------------------
                epoch += 1
                config.train.epoch = epoch
                if psnr_avg > max_psnr:
                    save_checkpoint(model, epoch, config.path.checkpoint, psnr_avg)
                    max_psnr = psnr_avg
                print("~" * 100)

                # dist.barrier()
                # if gpu_id == 0:
                # evaluation(model, eval_loader, config)
                if epoch == config.train.num_epochs:
                    raise Exception(f'epoch {epoch} >= max epoch {config.train.num_epochs}')
                time_start = time.time()
                print(f'Epoch={epoch}, lr={optimizer.param_groups[0]["lr"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', type=str, default='./options/ovsr.yml')
    cfg = parser.parse_args()

    with open(cfg.options, 'r', encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file.read())
        config = DICT2OBJ(config)

    gpu_id = config.gpu_id
    print("gpu_id = ", gpu_id)
    config.seed = random.randint(1, 10000)

    config.path.checkpoint = join(config.path.base, config.path.checkpoint, config.model.name)
    config.path.eval_result = join(config.path.base, config.path.eval_result, config.model.name)
    config.path.resume = join(config.path.checkpoint, f'{config.train.resume:04}.pth')
    config.path.eval_file = join(config.path.base, f'eval_{config.model.name}.txt')

    automkdir(config.path.checkpoint)
    automkdir(config.path.eval_result)

    config.network = importlib.import_module(f'models.{config.model.file}').Net(config)

    if config.function == 'get_complexity':
        macs, params = get_model_complexity_info(config.network, (3, 1, 180, 320), as_strings=False,
                                                 print_per_layer_stat=False, verbose=True,
                                                 ignore_modules=[torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.PReLU])
        print('Computational complexity: {:,}'.format(macs))
        print('Number of parameters: {:,}'.format(params))
        exit()

    function = getattr(sys.modules[__name__], config.function)

    # print("function: ", function)

    train(gpu_id, config)
