import random
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse
import sys
import os
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cfl import CFL_ConvBlock
from dataloader import get_concat_dataloader
from utils import mkdir_if_missing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.resnet import *
from models.densenet import densenet121

_model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'densenet121': densenet121
}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--t1_ckpt", type=str, required=True)
    parser.add_argument("--t2_ckpt", type=str, required=True)

    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--only_kd", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=60)
    return parser

def get_samples(loader, classes, sample_num=10):
    class_sample_num = { c: 0 for c in classes }
    samples = []
    samples_lbl = []
    finished_class = 0
    for itr_cnt, (images, labels) in enumerate(loader):
        for img, lbl in zip(images, labels):
            lbl = int(lbl.numpy())
            if lbl in classes and class_sample_num[lbl]<sample_num:
                samples.append(img)
                samples_lbl.append(lbl)
                class_sample_num[lbl]+=1

                if class_sample_num[lbl]==sample_num:
                    finished_class+=1
                    if finished_class==len(classes):
                        return torch.stack( samples, dim=0 ), torch.from_numpy( np.array(samples_lbl) )
    return torch.stack( samples, dim=0 ), torch.from_numpy( np.array(samples_lbl) )

def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ckpt_dir = './checkpoints'
    # Set up dataloader
    train_loader, val_loader = get_concat_dataloader(data_root=opts.data_root, batch_size=64, download=opts.download)
    
    t1_ckpt = opts.t1_ckpt
    t2_ckpt = opts.t2_ckpt
    stu_ckpt = opts.ckpt

    t1_model_name = t1_ckpt.split('/')[1].split('_')[1]
    t2_model_name = t2_ckpt.split('/')[1].split('_')[1]
    stu_model_name = stu_ckpt.split('/')[1].split('_')[1]

    num_classes = 200+120
    
    # setup networks
    t1 = _model_dict[t1_model_name](num_classes=200).to(device)
    t2 = _model_dict[t2_model_name](num_classes=120).to(device)
    stu = _model_dict[stu_model_name](num_classes=num_classes).to(device)

    print("Loading pretrained teachers, T1: %s, T2: %s"%(t1_model_name, t2_model_name))
    t1.load_state_dict(torch.load(t1_ckpt, map_location=lambda storage, loc: storage)['model_state'])
    t2.load_state_dict(torch.load(t2_ckpt, map_location=lambda storage, loc: storage)['model_state'])

    print("Loading student: %s"%(stu_model_name))
    stu.load_state_dict( torch.load(stu_ckpt, map_location=lambda storage, loc: storage)['model_state'] )

    # setup cfl block
    t1_feature_dim = t1.fc.in_features
    t2_feature_dim = t2.fc.in_features
    is_densenet = True if 'densenet' in stu_model_name else False
    if is_densenet:
        stu_feature_dim = stu.classifier.in_features
    else:
        stu_feature_dim = stu.fc.in_features
    
    def forward_hook(module, input, output):
        module.output = output # keep feature maps

    t1.layer4.register_forward_hook(forward_hook)
    t2.layer4.register_forward_hook(forward_hook)

    if is_densenet:
        stu.features.register_forward_hook(forward_hook)
    else:
        stu.layer4.register_forward_hook(forward_hook)
    
    teachers = [t1, t2]

    tsne_on_features_space = True if ( t1_feature_dim==t2_feature_dim and t2_feature_dim==stu_feature_dim) else False
    cfl_blk = CFL_ConvBlock(stu_feature_dim, [t1_feature_dim, t2_feature_dim], 128).to(device)
    cfl_blk.load_state_dict(torch.load(stu_ckpt, map_location=lambda storage, loc: storage)['cfl_state'])

    t1.eval()
    t2.eval()
    stu.eval()
    cfl_blk.eval()

    mkdir_if_missing('tsne_results')
    mkdir_if_missing('tsne_results/%s'%stu_model_name)
    
    marker_list = [ '^', 's' ]
    with torch.no_grad():
        # it will draw 15 tsne with different class split.
        for j in range(15):
            print("Split %d/15"%j)
            print('[Common Space]Collecting samples ...')
            class_list = np.arange(j, num_classes, 16) # (120+200) // 20 classes = 16 (interval)
            cmap = matplotlib.cm.get_cmap('tab20')
            # TODO: make it fast.
            images, labels = get_samples(val_loader, class_list, 50)   
            sample_class_num = len(class_list)
            print("%d samples selected"%len(images))

            print("[Common Space]Extracting features ...")
            ft1 = []
            ft2 = []
            fs = []
            hs = []
            ht1 = []
            ht2 = []

            for img, lbl in tqdm(zip(images, labels)):
                img = img.unsqueeze(0).to(device, dtype=torch.float)
                lbl = lbl.unsqueeze(0).to(device, dtype=torch.long)

                _ = t1(img)
                _ = t2(img)
                _ = stu(img)

                _ft1 = t1.layer4.output
                _ft2 = t2.layer4.output
                if is_densenet:
                    _fs = stu.features.output
                else:
                    _fs = stu.layer4.output
                (_hs, _ht), (_, _) = cfl_blk(_fs, [_ft1, _ft2])

                ft1.append(_ft1)
                ft2.append(_ft2)
                fs.append(_fs)
                hs.append(_hs)
                ht1.append(_ht[0])
                ht2.append(_ht[1])
                
            ft1 = torch.cat(ft1, dim=0)
            ft2 = torch.cat(ft2, dim=0)
            fs = torch.cat(fs, dim=0)
            hs = torch.cat(hs, dim=0)
            ht = [ torch.cat(ht1, dim=0), torch.cat(ht2, dim=0) ]

            # visualize the last cfl space
            #ft = [ t1.layer4.output, t2.layer4.output ]
            #fs = stu.layer4.output
            # get common feature hs and ht
            
            N, C, H, W = hs.shape
            features = [ hs.detach().view(N, -1) ]
            for ht_i in ht:
                features.append(ht_i.detach().view(N,-1))

            # The pretrained model use GAP to get 1D features. Here we also pool the common feature to make it clustered.
            features = F.normalize( torch.cat( features, dim=0 ), p=2, dim=1 ).view(3*N, C, -1).mean(dim=2).cpu().numpy()
            print("[Common Space] TSNE ...")
            tsne_res = TSNE(n_components=2, random_state=23333).fit_transform( features )
            print("[Common Space] TSNE finished ...")

            print("[Common Space] Ploting ... ")
            fig = plt.figure(1,figsize=(10,10))
            plt.axis('off')
            ax = fig.add_subplot(1, 1, 1)
            
            step_size = 1.0/sample_class_num 
            labels = labels.detach().cpu().numpy()
            
            label_to_color = { class_list[i]: cmap( step_size*i ) for i in range(sample_class_num) }
            sample_to_color = [ label_to_color[labels[i]] for i in range(len(labels)) ]
            ax.scatter(tsne_res[:N,0], tsne_res[:N, 1], c=sample_to_color, label = 'stu', marker="o", s = 30)
            
            for i in range(2): # 2 classification tasks
                ax.scatter(tsne_res[(i+1)*N:(i+2)*N,0], tsne_res[(i+1)*N:(i+2)*N, 1], c='',edgecolors=sample_to_color, label = 't%d'%i, marker=marker_list[i], s = 30)
            ax.legend(fontsize="xx-large", markerscale=2)
            plt.show()
            plt.savefig('tsne_results/%s/common_space_tsne_%d.png'%(stu_model_name, j))
            plt.close()

            # ========= Draw features space =========
            if tsne_on_features_space: 
                features = [ fs, ft1, ft2 ]
                features = F.normalize( torch.cat( features, dim=0 ), p=2, dim=1 ).view(3*N, C, -1).mean(dim=2).cpu().numpy()
                print("[Feature Space] TSNE ...")
                tsne_res = TSNE(n_components=2, random_state=23333).fit_transform( features )
                print("[Feature Space] TSNE finished ...")
    #
                print("[Feature Space] Ploting ... ")
                fig = plt.figure(1,figsize=(10,10))
                plt.axis('off')
                ax = fig.add_subplot(1, 1, 1)
                
                step_size = 1.0/sample_class_num 
                #labels = labels.detach().cpu().numpy()
                label_to_color = { class_list[i]: cmap( step_size*i ) for i in range(sample_class_num) }
                sample_to_color = [ label_to_color[labels[i]] for i in range(len(labels)) ]
                ax.scatter(tsne_res[:N,0], tsne_res[0:N, 1], c=sample_to_color, label = 'stu', marker="o", s = 30)
                
                for i in range(2): # 2 classification tasks
                    ax.scatter(tsne_res[(i+1)*N:(i+2)*N,0], tsne_res[(i+1)*N:(i+2)*N, 1], c='',edgecolors=sample_to_color, label = 't%d'%i, marker=marker_list[i], s = 30)
                ax.legend(fontsize="xx-large", markerscale=2)
                plt.show()
                plt.savefig('tsne_results/%s/feature_space_tsne_%d.png'%(stu_model_name, j))
                plt.close()


if __name__ == '__main__':
    main()
