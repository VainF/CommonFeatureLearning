import os
from torch.utils import data
from torchvision import transforms
from datasets import CUB200, StanfordDogs

def get_concat_dataloader(data_root, batch_size=64, download=False):
    transforms_train = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    transforms_val = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    cub_root = os.path.join(data_root, 'cub200')
    train_cub = CUB200(root=cub_root, split='train',
                        transforms=transforms_train,
                        download=download, offset=0)
    val_cub = CUB200(root=cub_root, split='test',
                        transforms=transforms_val,
                        download=False, offset=0)
    dogs_root = os.path.join(data_root, 'dogs')
    train_dogs = StanfordDogs(root=dogs_root, split='train',
                                transforms=transforms_train,
                                download=download, offset=200)
    val_dogs = StanfordDogs(root=dogs_root, split='test',
                            transforms=transforms_val,
                            download=False, offset=200) # add offset
    train_dst = data.ConcatDataset([train_cub, train_dogs])
    val_dst = data.ConcatDataset([val_cub, val_dogs])

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4)
    return train_loader, val_loader