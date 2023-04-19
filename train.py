from utils.dataset import MyDataset
from torch.utils.data import DataLoader
from models.myNN import MyNNModel
import pandas as pd

dataset = MyDataset('data/train.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(next(iter(dataloader))[0].shape[1])
# model = MyNNModel(dataset.data[0].shape[0])
