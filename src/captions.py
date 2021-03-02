import os
import pandas as pd

from utils import Vocab

vocab = Vocab()
vocab.set_min_count(3)

# loop over the captions file and create a dataframe
# with image and its corresponding captions

# create a dictionary of imageid -> list(captions)
dict_img_captions = {}
with open("..\data\captions.txt") as cap_file:
    for ix, line in enumerate(cap_file):
        print(ix)
        if ix == 0:
            # skip file header line
            continue

        img, cap = line.strip().split(",", 1)  # split the line on 1st comma
        # print(ix)
        caption = _vocab.preprocess_text(cap, add_to_vocab=True)
        if img in dict_img_captions:
            # image already in the dictionary
            dict_img_captions[img].append(cap)
        else:
            # add new image to dict
            dict_img_captions[img] = [cap]
        # break
print('Maximum caption length: ', _vocab.MAX_LEN)
_vocab.store_json()

# create a dataframe from the dict
df_captions = pd.DataFrame.from_dict(dict_img_captions, orient="index").reset_index()
df_captions.rename(columns={'index': 'image_name',
                            0: 'caption_1',
                            1: 'caption_2',
                            2: 'caption_3',
                            3: 'caption_4',
                            4: 'caption_5',
                            }, inplace=True)

# save this as a new csv file to avoid redo
df_captions.to_csv("../data/flickr8k_img_captions.csv", index=False)
