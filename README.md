# Common Feature Learning
Unofficial implementation of [Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning](http://arxiv.org/abs/1906.10546) (*IJCAI 2019*)

## Results

### Teacher Performance
Teacher Model   |        Dataset     |    num_classes       |    Acc   
:--------------:|:------------------:|:--------------------:| :-----------:
ResNet18        |   [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200.html)           |     200              |    0.7411
ResNet34        |   [Stanford-Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)    |     120              |    0.8663

### Student Performance
Target Model    |    KD       |       CFL   
:--------------:|:-----------:|:-------------------:
ResNet34        |   0.7679    |      **0.7762**
ResNet50        |   0.7909    |      **0.7926** 
DenseNet121     |   0.7769    |      **0.7805**

### Accuracy Curve
<div>
<img src="logs/acc-resnet34.png" width="30%">
<img src="logs/acc-resnet50.png" width="30%">
<img src="logs/acc-densenet121.png" width="30%">
</div>

## TSNE visualization on 20 classes dogs+birds

Features in Feature Space are extracted directly from models (e.g. output of resnet layer4).   
The following is TSNE results of two space. Samples are selected from StanfordDogs & CUB200 datasets.   
see `amal.py` and `tsne_common_space.py` for more information.

Target Model   |  Common Space             |  Feature Space
:----------:|:-------------------------:|:-------------------------:
ResNet34  | ![cfl-feature-space](tsne_results/resnet34/common_space_tsne_0.png)  |  ![cfl-feature-space](tsne_results/resnet34/feature_space_tsne_0.png)
ResNet50  |  ![cfl-feature-space](tsne_results/resnet50/common_space_tsne_0.png) |   None
DenseNet121  |  ![cfl-feature-space](tsne_results/densenet121/common_space_tsne_0.png) |   None

## Quick Start
This example shows how to extract common features from a bird classifier (ResNet34) and a dog classifier (ResNet18).

### 1. Download Datasets
```bash
python download_data.py
```

### 2. Get trained Teachers

[Download Link]()

### 3. Run
```bash
python amal.py --model resnet34 --gpu_id 0
```
or
```bash
bash run_all.sh
```


