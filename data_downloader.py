import os
import torch
from data import load_init_data
import pathlib

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_path = './dataset/'
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True) 
    load_init_data('mnist', device,True, data_path)
    load_init_data('cifar10', device,True, data_path)

if __name__ == "__main__":
    main()
