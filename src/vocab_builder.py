import os
import pandas as pd
from tqdm import trange, tqdm
from time import sleep

from src.utils import VocabBuilder

vocab = VocabBuilder(min_count=2, json_path='data/vocab.json')

# loop over the captions file and create a dataframe
# with image and its corresponding captions
file_path = os.path.abspath("data\captions.txt")

dict_img_captions = {}
with open(file_path) as (cap_file):
    for i, line in enumerate(tqdm(cap_file)):
        if i == 0:
            continue    # skip file header line

        img, cap = line.strip().split(",", 1)  # split the line on 1st comma
        caption = vocab.add_to_vocab(cap)
        dict_img_captions[img] = [cap]

print('Maximum caption length : ', vocab.MAX_LEN)
vocab.store_json()

# create a dataframe from the dict
df_captions = pd.DataFrame.from_dict(dict_img_captions, orient="index").reset_index()
df_captions.rename(columns={'index': 'image_name', 0: 'caption'}, inplace=True)

# save this as a new csv file to avoid redo
df_captions.to_csv(os.path.abspath("data/flickr8k_img_captions_new.csv"), index=False)