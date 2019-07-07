import random
import os, sys
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils import data

from loss import SoftCELoss, CFLoss
from utils.stream_metrics import StreamClsMetrics, AverageMeter
from models.cfl import CFL_ConvBlock
from datasets import StanfordDogs, CUB200
from utils import mkdir_if_missing, Logger
from dataloader import get_concat_dataloader
from torchvision import transforms
from models.resnet import *
from models.densenet import *

_model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'densenet121': densenet121
}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default='resnet34')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cfl_lr", type=float, default=None)

    parser.add_argument("--t1_ckpt", type=str, default='checkpoints/cub200_resnet18_best.pth')
    parser.add_argument("--t2_ckpt", type=str, default='checkpoints/dogs_resnet34_best.pth')

    return parser

def amal(cur_epoch, criterion_ce, criterion_cf, model, cfl_blk, teachers, optim, train_loader, device, scheduler=None, print_interval=10):
    """Train and return epoch loss"""
    t1, t2 = teachers

    if scheduler is not None:
        scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
    avgmeter = AverageMeter()
    is_densenet = isinstance(model, DenseNet)
    for cur_step, (images, labels) in enumerate(train_loader):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # get soft-target logits
        optim.zero_grad()
        with torch.no_grad():
            t1_out = t1(images)
            t2_out = t2(images)
            t_outs = torch.cat((t1_out, t2_out), dim=1)

            ft1 = t1.layer4.output
            ft2 = t2.layer4.output
        
        # get student output
        s_outs = model(images)
        if is_densenet:
            fs = model.features.output
        else:
            fs = model.layer4.output

        ft = [ft1, ft2]

        (hs, ht), (ft_, ft) = cfl_blk(fs, ft)

        loss_ce = criterion_ce(s_outs, t_outs)
        loss_cf = 10*criterion_cf(hs, ht, ft_, ft)

        loss = loss_ce + loss_cf
        loss.backward()
        optim.step()

        avgmeter.update('loss', loss.item())
        avgmeter.update('interval loss', loss.item())
        avgmeter.update('ce loss', loss_ce.item())
        avgmeter.update('cf loss', loss_cf.item())
        
        if (cur_step+1) % print_interval == 0:
            interval_loss = avgmeter.get_results('interval loss')
            ce_loss = avgmeter.get_results('ce loss')
            cf_loss = avgmeter.get_results('cf loss')

            print("Epoch %d, Batch %d/%d, Loss=%f (ce=%f, cf=%s)" %
                  (cur_epoch, cur_step+1, len(train_loader), interval_loss, ce_loss, cf_loss))
            avgmeter.reset('interval loss')
            avgmeter.reset('ce loss')
            avgmeter.reset('cf loss')
    return avgmeter.get_results('loss')


def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)

            preds = outputs.detach()  # .max(dim=1)[1].cpu().numpy()
            targets = labels  # .cpu().numpy()

            metrics.update(preds, targets)
        score = metrics.get_results()
    return score


def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set up random seed
    mkdir_if_missing('checkpoints')
    mkdir_if_missing('logs')
    sys.stdout = Logger(os.path.join('logs', 'amal_%s.txt'%(opts.model)))
    print(opts)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    cur_epoch = 0
    best_score = 0.0

    mkdir_if_missing('checkpoints')
    latest_ckpt = 'checkpoints/amal_%s_latest.pth'%opts.model
    best_ckpt = 'checkpoints/amal_%s_best.pth'%opts.model

    #  Set up dataloader
    train_loader, val_loader = get_concat_dataloader(data_root=opts.data_root, batch_size=opts.batch_size, download=opts.download)

    # pretrained teachers
    t1_model_name = opts.t1_ckpt.split('/')[1].split('_')[1]  
    t1 = _model_dict[t1_model_name](num_classes=200).to(device) # cub200
    t2_model_name = opts.t2_ckpt.split('/')[1].split('_')[1]  
    t2 = _model_dict[t2_model_name](num_classes=120).to(device) # dogs
    print("Loading pretrained teachers ...\nT1: %s, T2: %s"%(t1_model_name, t2_model_name))
    t1.load_state_dict(torch.load(opts.t1_ckpt)['model_state'])
    t2.load_state_dict(torch.load(opts.t2_ckpt)['model_state'])
    t1.eval()
    t2.eval()

    print("Target student: %s"%opts.model)
    stu = _model_dict[opts.model](pretrained=True, num_classes=120+200).to(device)
    metrics = StreamClsMetrics(120+200)

    # Setup Common Feature Blocks
    t1_feature_dim = t1.fc.in_features
    t2_feature_dim = t2.fc.in_features

    is_densenet = True if 'densenet' in opts.model else False

    if is_densenet:
        stu_feature_dim = stu.classifier.in_features
    else:
        stu_feature_dim = stu.fc.in_features
    
    cfl_blk = CFL_ConvBlock(stu_feature_dim, [t1_feature_dim, t2_feature_dim], 128).to(device)

    def forward_hook(module, input, output):
        module.output = output # keep feature maps

    t1.layer4.register_forward_hook(forward_hook)
    t2.layer4.register_forward_hook(forward_hook)

    if is_densenet:
        stu.features.register_forward_hook(forward_hook)
    else:
        stu.layer4.register_forward_hook(forward_hook)

    params_1x = []
    params_10x = []
    for name, param in stu.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    cfl_lr = opts.lr*10 if opts.cfl_lr is None else opts.cfl_lr
    optimizer = torch.optim.Adam([{'params': params_1x,             'lr': opts.lr},
                                  {'params': params_10x,            'lr': opts.lr*10},
                                  {'params': cfl_blk.parameters(),  'lr': cfl_lr} ],
                                 lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1)

    # Loss
    criterion_ce = SoftCELoss(T=1.0)
    criterion_cf = CFLoss(normalized=True)

    def save_ckpt(path):
        """ save current model
        """
        state = {
            "epoch": cur_epoch,
            "model_state": stu.state_dict(),
            "cfl_state": cfl_blk.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }
        torch.save(state, path)
        print("Model saved as %s" % path)

    print("Training ...")
    # ===== Train Loop =====#
    while cur_epoch < opts.epochs:
        stu.train()
        epoch_loss = amal(cur_epoch=cur_epoch,
                            criterion_ce=criterion_ce,
                            criterion_cf=criterion_cf,
                            model=stu,
                            cfl_blk=cfl_blk,
                            teachers=[t1, t2],
                            optim=optimizer,
                            train_loader=train_loader,
                            device=device,
                            scheduler=scheduler)
        print("End of Epoch %d/%d, Average Loss=%f" %
              (cur_epoch, opts.epochs, epoch_loss))

        # =====  Latest Checkpoints  =====
        save_ckpt(latest_ckpt)
        # =====  Validation  =====
        print("validate on val set...")
        stu.eval()
        val_score = validate(model=stu,
                             loader=val_loader,
                             device=device,
                             metrics=metrics)
        print(metrics.to_str(val_score))
        sys.stdout.flush()
        # =====  Save Best Model  =====
        if val_score['Overall Acc'] > best_score:  # save best model
            best_score = val_score['Overall Acc']
            save_ckpt(best_ckpt)
        cur_epoch += 1

if __name__ == '__main__':
    main()
