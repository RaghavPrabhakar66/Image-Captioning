import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import Flickr8k, vocab
from old_utils import show_imgs

captions_csv = pd.read_csv(os.path.abspath(r"data\flickr8k_img_captions.csv"))

# split the dataframe into train and testget_word_embedding
train, val = train_test_split(captions_csv, test_size=0.2, random_state=12)
# split the test set into test and validation
val, test = train_test_split(val, test_size=0.1, random_state=12)

my_transforms_train = transforms.Compose([
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            transforms.Resize((512, 512)),
                                            transforms.ToTensor()
    ])

train_set = Flickr8k(df=train,
                    data_dir=os.path.abspath('data'), transforms=my_transforms_train)

train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(train_dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['captions'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        for idx, caption_set in enumerate(sample_batched['captions']):
            for caption in caption_set:
                for token in caption.tolist():
                    print(vocab.get_word_token(token), end=" ")
                print()
            show_imgs(sample_batched['image'][idx])
        break