from datasets import CUB200, StanfordDogs

CUB200(root='./data/cub200', split='train',
                    download=True)
StanfordDogs(root='./data/dogs', split='train',
                    download=True)