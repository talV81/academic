from torch.utils.data import Dataset
from scipy import ndimage
from .augmentation import augmentation
import skimage
import imageio
import numpy as np
import h5py
import os
import random

class HPADataset(Dataset):

    def __init__(self, data_path, phase='train', transform=False, max_mean="max", target_channels="3"):
        """Custom PyTorch Dataset for hpa dataset

        Parameters
        ----------
            data_path: str
                path to the nuclei dataset hdf5 file
            phase: str, optional
                phase this dataset is used for (train, val. test)
        """

        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.max_mean = max_mean

        if "," in target_channels:
            self.target_channels = [int(c) for c in target_channels.split(',')]
        else:
            self.target_channels = [int(target_channels)]

        self.target_dim = len(self.target_channels)


        with h5py.File(self.data_path,"r") as h:
            self.data_names = list(h['/raw']['train'].keys())

            self.dim = 1    # decision to only use one channel (rgb are the same, a is all 1s)


    def __len__(self):

        return len(self.data_names)


    def __getitem__(self, idx):

        with h5py.File(self.data_path,"r") as h:
            data = h['/raw']['train'][self.data_names[idx]][:]

        data = np.transpose(data, (2,0,1))
        data = data / 255.
        
        y = data[self.target_channels].copy()

        y = np.expand_dims(y, 0)
        
        if self.max_mean == "max":
            x = np.max(data, axis=0)
        else:
            x = np.mean(data, axis=0)
        x = np.expand_dims(x, 0)

        if self.transform: 
            x,y = augmentation(x,y)
        
        return x, y
