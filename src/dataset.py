import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class Flickr8k(Dataset):
    def __init__(self, df, data_dir=None, transforms=None):
        
        self.data = df
        self.data_dir = data_dir
        self.images = os.path.join(self.data_dir, 'Images')
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # iloc is index based, which is what we need here
        img_name = os.path.join(self.images, self.data.iloc[idx][0])
        image = Image.open(img_name)

        caption = list(self.data.iloc[idx][1:])

        if self.transforms:
            image = self.transforms(image)

        return {'image': image, 'caption': torch.tensor(caption), 'debug_img_name': img_name, 'deubg_idx': idx}
