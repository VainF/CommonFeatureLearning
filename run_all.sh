python amal.py --gpu_id 3 --model resnet34 --lr 1e-4
python kd.py --gpu_id 3 --model resnet34 --lr 1e-4

python amal.py --gpu_id 3 --model resnet50 --lr 5e-5
python kd.py --gpu_id 3 --model resnet50 --lr 5e-5

python amal.py --gpu_id 3 --model densenet121 --lr 2e-4
python kd.py --gpu_id 3 --model densenet121 --lr 2e-4