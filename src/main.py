import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary

from dataset import Flickr8k, vocab
from models import Model
from utils import show_imgs
from train import train_fit, validation_fit

def main(params, show_imgs=False, resume_training=False):
    
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
            embed_size=params['embed_size'], 
            hidden_size=params['hidden_size'], 
            vocab_size=vocab.MAX_INDEX, 
            lstm_cells=params['lstm_cells'], 
            lstm_dropout=params['lstm_dropout'],
            verbose=False,
            device=params['device'])

    if params['optimizer']=='sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], betas=params['betas'], eps=params['eps'], weight_decay=params['weight_decay'], decay=1e-5, decay=1e-5, momentum=0.9, nesterov=True)
    elif params['optimizer']=='rmsprop':
      optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], betas=params['betas'], eps=params['eps'], weight_decay=params['weight_decay'], decay=1e-5)
    elif params['optimizer']=='adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=params['betas'], eps=params['eps'], weight_decay=params['weight_decay'])
    elif params['optimizer']=='nadam':
      optimizer = torch.optim.Nadam(model.parameters(), lr=params['lr'], betas=params['betas'], eps=params['eps'], weight_decay=params['weight_decay'], decay=1e-5)

    criterion = torch.nn.CrossEntropyLoss()

    train_loss , train_accuracy = [], []
    val_loss , val_accuracy = [], []
    start_epoch = 0

    if resume_training:
        loaded_checkpoint = torch.load(params['LOAD_CKPT_PATH'])

        model.load_state_dict(loaded_checkpoint['state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer'])

        start_epoch = loaded_checkpoint['epoch']

        train_loss = loaded_checkpoint['training_loss']
        train_accuracy = loaded_checkpoint['training_acc']

        val_loss = loaded_checkpoint['val_loss']
        val_accuracy = loaded_checkpoint['val_acc']

    for epoch in range(start_epoch, params['epochs']):
        print(f"\nEpoch {epoch+1} of {params['epochs']}")
        print("-"*15)
        print()

        train_epoch_loss, train_epoch_accuracy = train_fit(params['device'], model, train_dataloader, optimizer, criterion, train_set)
        val_epoch_loss, val_epoch_accuracy     = validation_fit(params['device'], model, val_dataloader, optimizer, criterion, val_set)

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)

        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        print(f"Train Loss:\t {train_epoch_loss:.8f}, Train Acc:\t {train_epoch_accuracy:.8f}")
        print(f'Val Loss:\t {val_epoch_loss:.8f}, Val Acc:\t {val_epoch_accuracy:.8f}')

        # save model checkpoint
        checkpoint = {
            'epoch'         : epoch + 1,
            'state_dict'    : model.state_dict(),
            'optimizer'     : optimizer.state_dict(),
            'training_loss' : train_loss,
            'training_acc'  : train_accuracy,
            'val_loss'      : val_loss,
            'val_acc'       : val_accuracy,
        }

        torch.save(checkpoint, os.path.join(params['CKPT_DIR'], 'model.pt'))

if __name__ == '__main__':
    params = {
        'csv_filepath'  : r'data\captions.csv',
        'CKPT_DIR'      : r'C:\Users\ragha\Desktop\Projects\Image-Captioning\models',
        'LOAD_CKPT_PATH': '',

        'device'        : 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers'   : 4,             # simple rule 4*no.of gpu'
        
        'seed'          : 42,
        'IMG_SIZE'      : 224,

        'backbone'      : 'resnet50',    # ['vgg16', 'resnet18', 'resnet50', 'efficientnet', 'inception']
        'embed_size'    : 128, 
        'hidden_size'   : 128, 
        'lstm_cells'    : 128, 
        'lstm_dropout'  : 0.5,

        'epochs'        : 2,
        'batch_size'    : 4,
        'optmizier'     : 'adam',         # ['adam', 'sgd', 'rmsprop' 'nadam']
        'lr'            : 0.01,
        'betas'         : (0.9, 0.999),
        'eps'           : 1e-8,
        'weight_decay'  : 0.0005,
    }

    pprint(params)

    main(
        params=params,
        show_imgs=False,
        resume_training=False
        )
