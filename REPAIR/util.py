from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader

class DatasetSplitter:
    def __init__(self, dataset, indices):
        self.indices = indices
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample, label = self.dataset[self.indices[idx]]

        return sample, label


class ModifyLabels(Dataset):
    def __init__(self, original_dataset, transform):
        self.original_dataset = original_dataset
        self.transform()

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample, label = self.original_dataset[idx]
        # Multiply the label by 2
        label *= 2
        return sample, label
    

class OffsetLabel:
    def __init__(self, offset):
        self.offset = offset

    def __call__(self, label):
        label = label + self.offset

        return label