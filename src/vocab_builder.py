from operator import index
import pprint
import os
import pandas as pd
from tqdm import tqdm
from utils import VocabBuilder, Vocab

#python -m spacy download en_core_web_sm

def main(create_vocab = False, process_captions = False, file_path = os.path.abspath("data\captions.csv"), min_count=6, json_path='data/vocab_with_6.json', final_csv_path='data/processed.csv'):
    if create_vocab:
        # loop over the captions file and create a dataframe
        # with image and its corresponding captions
        vocab = VocabBuilder(min_count=min_count, json_path=json_path)

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
        captions = {}
        final_img_name, final_caption, references = [], [], []

        # Use vocabulary generated to tokenize captions and store them in a seperate file
        tokenize = Vocab().get_word_embedding
        cap_file = pd.read_csv(file_path)

        for i in tqdm(cap_file.index):
            new_line = tokenize(cap_file.loc[i][1])
            if cap_file.loc[i][0] in captions:
                captions[cap_file.loc[i][0]].append(new_line)
            else:
                captions[cap_file.loc[i][0]] = [new_line]

        for i, img in enumerate(captions.keys()):
            for j in range(5):
                temp = captions[img][:]
                temp.pop(j)
                final_img_name.append(img)
                final_caption.append(captions[img][j])
                references.append(temp)

        captions_df = pd.DataFrame(list(zip(final_img_name, final_caption, references)))
        print(len(captions_df))
        captions_df.to_csv(final_csv_path, header=False, index=False)

if __name__=='__main__':
    main(
        create_vocab     = False,
        process_captions = True,
        file_path        = os.path.abspath("data/captions.csv"),
        min_count        = 6,
        json_path        = 'data/vocab_with_6.json',
        final_csv_path   = 'data/processed_bleu.csv'
    )