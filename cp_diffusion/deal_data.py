# _*_coding:utf-8_*_
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.Data = data
        self.Label = label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index]).unsqueeze(0)
        label = torch.Tensor(self.Label[index]).unsqueeze(0)
        return data, label

def load_data(file_path, batch_size):
    with open(file_path, 'rb') as file:
        datas = pickle.load(file)
    data = np.asarray(datas['promoter'])
    label = np.asarray(datas['expression'])

    dataset = MyDataset(data, label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
