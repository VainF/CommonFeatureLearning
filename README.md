# Common Feature Learning
Official implementation of [Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning](http://arxiv.org/abs/1906.10546) (*IJCAI 2019*) in pytorch.

<div align="center">
<img src="CFL.png" width=40%>
</div>

## Results

### Teacher Performance
Teacher Model   |        Dataset     |    num_classes       |    Acc   
:--------------:|:------------------:|:--------------------:| :-----------:
ResNet18        |   [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200.html)        |     200              |    0.7411
ResNet34        |   [StanfordDogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)     |     120              |    0.8663

### Student Performance (CUB200+StanfordDogs)
Target Model    |    KD       |       CFL   
:--------------:|:-----------:|:-------------------:
ResNet34        |   0.7684    |      **0.7721**
ResNet50        |   0.7965    |      **0.7997** 
DenseNet121     |   0.7769    |      **0.7815**

see *logs* for more information

### Accuracy Curve
<div>
<img src="logs/acc-resnet34.png" width="30%">
<img src="logs/acc-resnet50.png" width="30%">
<img src="logs/acc-densenet121.png" width="30%">
</div>

## TSNE Visualization of 20 Classes

`Feature Space`: space constructed with intermediate outputs.  
`Common Space`: common feature space in CFL Blocks.

**Some Feature spaces are None because of different feature dimensions (e.g. 2048 for ResNet50 but 512 for ResNet34)**

Target Model   |  Common Space             |  Feature Space
:----------:|:-------------------------:|:-------------------------:
ResNet34  | ![cfl-feature-space](tsne_results/resnet34/common_space_tsne_1.png)  |  ![cfl-feature-space](tsne_results/resnet34/feature_space_tsne_1.png)
ResNet50  |  ![cfl-feature-space](tsne_results/resnet50/common_space_tsne_1.png) |   None
DenseNet121  |  ![cfl-feature-space](tsne_results/densenet121/common_space_tsne_1.png) |   None

## Quick Start

#### 1. Download Datasets
```bash
python download_data.py
```

#### 2. Get Trained Teacher Models
ResNet18 & ResNet34, 242.9 MB  
[Google Drive](https://drive.google.com/open?id=1cMOwJlf4S9fGwWv4hSXnpSwSrwGkGW94)  
[BaiDu Yun](https://pan.baidu.com/s/1qydtKOSn8vVUkqWjuyHblA)  

#### 3. Train
```bash
python amal.py --model resnet34 --gpu_id 0 --lr 1e-4 --cfl_lr 5e-4
python kd.py --model resnet34 --gpu_id 0 --lr 1e-4
```
or
```bash
bash run_all.sh
```

#### 4. Draw Accuracy Curve
```bash
cd logs/
python draw_acc_curve.py
```

#### 5. TSNE
TSNE results will be saved at tsne_results/MODEL_NAME/ 

```bash
# ResNet34
python tsne_common_space.py --ckpt checkpoints/amal_resnet34_best.pth --t1_ckpt checkpoints/cub200_resnet18_best.pth --t2_ckpt checkpoints/dogs_resnet34_best.pth --gpu_id 0

# ResNet50
python tsne_common_space.py --ckpt checkpoints/amal_resnet50_best.pth --t1_ckpt checkpoints/cub200_resnet18_best.pth --t2_ckpt checkpoints/dogs_resnet34_best.pth --gpu_id 0

# DenseNet121
python tsne_common_space.py --ckpt checkpoints/amal_densenet121_best.pth --t1_ckpt checkpoints/cub200_resnet18_best.pth --t2_ckpt checkpoints/dogs_resnet34_best.pth --gpu_id 0
```

## Citation
```
@inproceedings{luo2019knowledge,
  title={Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning},
  author={Luo, Sihui and Wang, Xinchao and Fang, Gongfan and Hu, Yao and Tao, Dapeng and Song, Mingli},
  booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2019},
}
```
