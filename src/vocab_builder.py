import os
from string import punctuation
import pandas as pd
from tqdm import tqdm
from utils import VocabBuilder, Vocab

def main(create_vocab, process_captions, vocab_path, cap_path, processed_path):
    if create_vocab:
        # loop over the captions file and create a dataframe
        # with image and its corresponding captions
        vocab = VocabBuilder(min_count=6, json_path=vocab_path)

        words = []
        with open(os.path.abspath(cap_path)) as (cap_file):
            maxlen = 0
            for i, line in enumerate(tqdm(cap_file)):
                if i == 0:
                    continue    # skip file header line
                
                # Add current caption to list of all the words
                _, cap_string = line.strip().split(",", maxsplit=1)  # split the line on 1st comma
                cap_string = ''.join([i for i in cap_string if i not in punctuation]).lower()
                cap_words = cap_string.split()
                words.extend(cap_words)
                
                cap_length = len(cap_words)
                if cap_length > maxlen:
                    maxlen = cap_length                

        vocab.generate_vocab(words, maxlen)                
        print('Maximum caption length : ', maxlen+2)

    if process_captions:
        # Use vocabulary generated to tokenize captions and store them in a seperate file
        tokenize = Vocab(vocab_path).get_word_embedding
        cap_file = pd.read_csv(cap_path)
        processed = []

        for i in tqdm(cap_file.index):
            new_line = tokenize(cap_file.loc[i][1])
            new_line.insert(0, cap_file.loc[i][0])
            processed.append(new_line)

        cap_file = pd.DataFrame().from_records(processed)
        cap_file.to_csv(processed_path, header=False, index=False)

if __name__=='__main__':
    main(
        create_vocab        = True,
        process_captions    = True,
        vocab_path          = 'data/Flickr30k/vocab.json',
        cap_path            = 'data/Flickr30k/captions.csv',
        processed_path      = 'data/Flickr30k/processed.csv'
    )