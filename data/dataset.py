import numpy as np
import os
import torch.utils.data as dataclass
class Displacement_dataset(dataclass.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.data = os.listdir(f'{root}/data')
        self.data = [f'{root}/data/{i}' for i in self.data]
        
        self.label = os.listdir(f'{root}/label')
        self.label = [f'{root}/label/{i}' for i in self.label]
        assert len(self.data) == len(self.label), 'data and label numbers are different.'

    def __getitem__(self, index):
        data  = np.load(self.data[index]).astype(np.float32)
        label = np.load(self.label[index]).astype(np.float32)
        data = self.transform(data)
        label = self.transform(label)
        return data, label
    
    def __len__(self):
        return len(self.data)
    
class Displacement_dataset_IQ(dataclass.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.data = os.listdir(f'{root}/data')
        self.data = [f'{root}/data/{i}' for i in self.data]
        
        self.I = os.listdir(f'{root}/I')
        self.I = [f'{root}/I/{i}' for i in self.I]
        
        self.Q = os.listdir(f'{root}/Q')
        self.Q = [f'{root}/Q/{i}' for i in self.Q]
        
        self.label = os.listdir(f'{root}/label')
        self.label = [f'{root}/label/{i}' for i in self.label]
        assert len(self.data) == len(self.label), 'data and label numbers are different.'

    def __getitem__(self, index):
        data  = np.load(self.data[index]).astype(np.float32)
        I  = np.load(self.I[index]).astype(np.float32)
        Q  = np.load(self.Q[index]).astype(np.float32)
        label = np.load(self.label[index]).astype(np.float32)
        data = self.transform(data)
        I = self.transform(I)
        Q = self.transform(Q)
        label = self.transform(label)
        return data, label, I, Q
    
    def __len__(self):
        return len(self.data)