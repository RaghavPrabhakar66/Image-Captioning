import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import Flickr8k, vocab
from models import Model
from utils import show_imgs

captions_csv = pd.read_csv(os.path.abspath(r"data\flickr8k_img_captions_new.csv"))

# split the dataframe into train and testget_word_embedding
train, val = train_test_split(captions_csv, test_size=0.2, random_state=12)
# split the test set into test and validation
val, test = train_test_split(val, test_size=0.1, random_state=12)

my_transforms_train = transforms.Compose([
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor()
    ])

train_set = Flickr8k(df=train,
                    data_dir=os.path.abspath('data'), transforms=my_transforms_train)

train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(train_dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['caption'].size())
    break
    # observe 4th batch and stop.
    if i_batch == 1:
        for idx, caption_set in enumerate(sample_batched['caption']):
            print(caption_set)
            for caption in caption_set:
                for token in caption.tolist():
                    print(token, end=" ")
                    #print(vocab.get_word_token(token), end=" ")
                print()
            show_imgs(sample_batched['image'][idx])
        break

model = Model(
            backbone='vgg16',
            embed_size=128, 
            hidden_size=128, 
            vocab_size=vocab.MAX_INDEX, 
            lstm_cells=128, 
            lstm_dropout=0.5)

#encoded_features = model.encoder.summary()

prediction = model(sample_batched)


"""
for caption in prediction:
    for token in caption.tolist():
        #print(token, end=" ")
        print(vocab.get_word_token(token), end=" ")
"""