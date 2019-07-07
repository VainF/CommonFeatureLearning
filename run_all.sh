# Uncomment to try different target models
# Train
# ResNet34
python amal.py --gpu_id 0 --model resnet34 --lr 1e-4 --cfl_lr 5e-4
python kd.py --gpu_id 0 --model resnet34 --lr 1e-4
python tsne_common_space.py --ckpt checkpoints/amal_resnet34_best.pth --t1_ckpt checkpoints/cub200_resnet18_best.pth --t2_ckpt checkpoints/dogs_resnet34_best.pth --gpu_id 0

# ResNet50
#python amal.py --gpu_id 0 --model resnet50 --lr 2e-5 --cfl_lr 2e-4
#python kd.py --gpu_id 0 --model resnet50 --lr 2e-5
#python tsne_common_space.py --ckpt checkpoints/amal_resnet50_best.pth --t1_ckpt checkpoints/cub200_resnet18_best.pth --t2_ckpt checkpoints/dogs_resnet34_best.pth --gpu_id 0

# DenseNet121
#python amal.py --gpu_id 0 --model densenet121 --lr 2e-4 --cfl_lr 2e-3
#python kd.py --gpu_id 0 --model densenet121 --lr 2e-4
#python tsne_common_space.py --ckpt checkpoints/amal_densenet121_best.pth --t1_ckpt checkpoints/cub200_resnet18_best.pth --t2_ckpt checkpoints/dogs_resnet34_best.pth --gpu_id 0
