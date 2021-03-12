import os
from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import wandb

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary

from src.dataset import Flickr8k
from src.models import Model
from src.utils import show_imgs, DiscordNotifier
from src.train import train_fit, validation_fit
from src.inference import Inference
from src.utils import Vocab

vocab = Vocab()

def main(params, show_imgs=False, resume_training=False, ignore=[]):
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(params['seed'])       # pytorch random seed
    np.random.seed(params['seed'])          # numpy random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    captions_csv = pd.read_csv(os.path.abspath(params['csv_filepath']))

    # split the dataframe into train and testget_word_embedding
    train, val = train_test_split(captions_csv, test_size=0.2, random_state=params['seed'])
    # split the test set into test and validation
    val, test = train_test_split(val, test_size=0.1, random_state=params['seed'])

    my_transforms_train = transforms.Compose([
                                                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                transforms.Resize((params['IMG_SIZE'], params['IMG_SIZE'])),
                                                transforms.ToTensor()
    ])

    my_transforms_val = transforms.Compose([
                                                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                transforms.Resize((params['IMG_SIZE'], params['IMG_SIZE'])),
                                                transforms.ToTensor()
    ])

    train_set = Flickr8k(df=train, data_dir=os.path.abspath('data'), transforms=my_transforms_train)
    train_dataloader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    val_set = Flickr8k(df=val, data_dir=os.path.abspath('data'), transforms=my_transforms_val)
    val_dataloader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    if show_imgs:
        for i_batch, sample_batched in enumerate(train_dataloader):
            print(i_batch, sample_batched['image'].size(), sample_batched['caption'].size())
            # observe 4th batch and stop.
            if i_batch == 2:
                for idx, caption_set in enumerate(sample_batched['caption']):
                    for caption in caption_set:
                        for token in caption.tolist():
                            print(token, end=" ")
                            #print(vocab.get_word_token(token), end=" ")
                        print()
                    show_imgs(sample_batched['image'][idx])
                break

    model = Model(
            backbone=params['backbone'],
            freeze_layers=params['freeze_layers'],
            embed_size=params['embed_size'], 
            hidden_size=params['hidden_size'], 
            vocab_size=vocab.MAX_INDEX, 
            lstm_cells=params['lstm_cells'], 
            lstm_dropout=0.5,
            verbose=False,
            device=params['device'])

    wandb.watch(model, log='all')

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=params['betas'], eps=params['eps'], weight_decay=params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    train_loss, val_loss = [], []
    start_epoch = 0

    for epoch in range(start_epoch, params['epochs']):
        train_epoch_loss, _ = train_fit(params['device'], model, train_dataloader, optimizer, criterion, train_set, ignore)
        val_epoch_loss, _     = validation_fit(params['device'], model, val_dataloader, optimizer, criterion, val_set)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)

        wandb.log({
            "Training Loss"      : train_epoch_loss,
            "Validation Loss"    : val_epoch_loss
        })

        print(f"Train Loss:\t {train_epoch_loss:.8f}")  
        print(f'Val Loss:\t {val_epoch_loss:.8f}')
