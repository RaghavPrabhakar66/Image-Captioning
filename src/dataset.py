import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import Vocab

tokenize = Vocab().get_word_embedding

class Flickr8k(Dataset):
    def __init__(self, df, data_dir=None, transforms=None):
        
        self.data = df
        self.data_dir = data_dir
        self.images = os.path.join(self.data_dir, 'Images')
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images, self.data.iloc[idx][0])
        image    = Image.open(img_name)

        caption    = self.data.iloc[idx][1]
        references = self.data.iloc[idx][2]

        if self.transforms:
            image = self.transforms(image)

        return {'image': image, 'caption': torch.tensor(tokenize(caption)), 'references': references}
