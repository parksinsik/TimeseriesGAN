import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader


class KospiDataset(Dataset):
    
    def __init__(self, file_path, normalize=True):
        
        data = pd.read_csv(file_path)
        
        data["time"] = data["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        data["date"] = data["time"].apply(lambda x: x.date())
        
        data.set_index("time", inplace=True)
        data.rename(columns={"29": "kospi200"}, inplace=True)
        
        data = data[data.index.hour != 18].dropna()
        data = data.resample(rule="5T").last().dropna()
        
        counts = data.groupby("date").count()
        remove_index = counts[counts["kospi200"] != 79].index
        
        data = data[~data["date"].isin(remove_index)].dropna()
        data.drop(columns=["date"], inplace=True)
        
        data = data.kospi200
        data = torch.from_numpy(np.expand_dims(np.array([group[1] for group in data.groupby(data.index.date)]), -1)).float()
        
        self.data = self.normalize(data) if normalize else data
        self.seq_length = data.size(1)
        
        org_deltas = data[:, -1] - data[:, 0] # 일자별 종가 - 시가
        self.org_deltas = org_deltas
        self.org_delta_max, self.org_delta_min = org_deltas.max(), org_deltas.min()
        
        norm_deltas = self.data[:, -1] - self.data[:, 0]
        self.norm_deltas = norm_deltas
        self.norm_delta_mean, self.norm_delta_std = norm_deltas.mean(), norm_deltas.std()
        self.norm_delta_max, self.norm_delta_min = norm_deltas.max(), norm_deltas.min()
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def normalize(self, x):
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min()) / (x.max() - x.min()) - 1) # -1 ~ 1
    
    def denormalize(self, x):
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception("You try denormalize, but inputs weren't normalized")
        return 0.5 * (x * self.max - x * self.min + self.max + self.min)
    
    def sample_deltas(self, n):
        return torch.randn(n, 1) * self.norm_delta_std + self.norm_delta_mean
    
    def normalize_deltas(self, x):
        return ((self.norm_delta_max - self.norm_delta_min) * (x - self.org_delta_min) / (self.org_delta_max - self.org_delta_min) + self.norm_delta_min)
