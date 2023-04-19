import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.drop(columns=['Name'])
        self.data = self.data.drop(columns=['Name'])
        self.data = self.data.drop(columns=['Name'])

        self.labels = self.data['label_column']
        self.data = self.data.drop(columns=['label_column'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx]), torch.tensor(self.y.iloc[idx])

    def get_num_features(self):
        return self.data.shape[1]


