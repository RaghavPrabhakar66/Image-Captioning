import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from old_utils import Vocab

vocab = Vocab()

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
        img_name = os.path.join(self.images, self.data.iloc[idx]['image_name'])
        image = Image.open(img_name)

        captions = np.array(self.data.iloc[idx:idx+1, 1:])[0].tolist()

        for i in range(len(captions)):
            captions[i] = vocab.get_word_embedding(captions[i])

        if self.transforms:
            image = self.transforms(image)

        return {'image': image, 'captions': torch.tensor(captions)}