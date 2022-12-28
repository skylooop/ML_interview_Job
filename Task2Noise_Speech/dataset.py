# DL libraries
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# linear algebra libraries
import numpy as np

import pandas as pd
# system libraries
import os
from pathlib import Path


class ClassificationSound(Dataset):
    
    def __init__(self, path_to_train: str):
        '''
        Specify path to training data. Folder that contains clean/noisy as subfolders
        '''
        super().__init__()
        self.path_to_train = path_to_train
        clean_class = os.listdir(os.path.join(self.path_to_train, 'clean/'))
        noisy_class = os.listdir(os.path.join(path_to_train, 'noisy/'))
        
        dir_clean = np.array([os.path.join(path_to_train, 'clean', speaker_id, speaker_sample) for speaker_id in sorted(clean_class) for speaker_sample in os.listdir(os.path.join(path_to_train, 'clean', speaker_id))])
        dir_noisy = np.array([os.path.join(path_to_train, 'noisy', speaker_id, speaker_sample) for speaker_id in sorted(noisy_class)for speaker_sample in os.listdir(os.path.join(path_to_train, 'clean', speaker_id))])

        self.dataframe = pd.DataFrame({'All_samples': np.hstack((dir_clean, dir_noisy)), 'labels': np.block([np.ones(len(dir_clean)), np.zeros(len(dir_noisy))])})
        
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        sample = np.load(self.dataframe['All_samples'].iloc[idx]).T
        
        # Resize and truncate to make same length
        processed_sample = self._process(sample)
        processed_sample = torch.Tensor(processed_sample).unsqueeze(0)
        return processed_sample, self.dataframe['labels'].iloc[idx]
        
    @staticmethod
    def _process(sample: np.ndarray) -> np.ndarray:
        '''
        Method for truncating/shifting sample
        '''
        
        if sample.shape[1] < 700:
            processed_sample = np.pad(sample, ((0, 0), (0, 700 - sample.shape[1])))
        else:
            processed_sample = sample[:, :700]
        return processed_sample
        
        
        