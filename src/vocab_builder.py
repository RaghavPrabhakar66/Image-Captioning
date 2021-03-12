from operator import index
import os
import pandas as pd
from tqdm import tqdm
from utils import VocabBuilder, Vocab

#python -m spacy download en_core_web_sm

def main(create_vocab = False, process_captions = False, file_path = os.path.abspath("data\captions.csv")):
    if create_vocab:
        # loop over the captions file and create a dataframe
        # with image and its corresponding captions
        vocab = VocabBuilder(min_count=6, json_path='data/vocab_with_6.json')

        dict_img_captions = {}
        with open(file_path) as (cap_file):
            for i, line in enumerate(tqdm(cap_file)):
                if i == 0:
                    continue    # skip file header line

                img, cap = line.strip().split(",", 1)  # split the line on 1st comma
                vocab.add_to_vocab(cap)
                
        print('Maximum caption length : ', vocab.MAX_LEN)
        vocab.store_json()

    if process_captions:
        # Use vocabulary generated to tokenize captions and store them in a seperate file
        tokenize = Vocab().get_word_embedding
        cap_file = pd.read_csv(file_path)
        processed = []
        -0
        for i in tqdm(cap_file.index):
            new_line = tokenize(cap_file.loc[i][1])
            new_line.insert(0, cap_file.loc[i][0])
            processed.append(new_line)
        
        cap_file = pd.DataFrame().from_records(processed)
        cap_file.to_csv('data/processed.csv', header=False, index=False)

if __name__=='__main__':
    main(
        create_vocab = True,
        process_captions = False,
        file_path = os.path.abspath("data/captions.csv")
    )