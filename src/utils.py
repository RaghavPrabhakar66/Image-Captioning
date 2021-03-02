import os
import spacy
import json
import matplotlib.pyplot as plt

sp = spacy.load("en_core_web_sm")  # load english core

PAD_token = 0   # Used for padding short sentences
SOS_token = 1   # Start-of-sentence token
EOS_token = 2   # End-of-sentence token
UNK_token = 3   # Unkown token
class Vocab:
    def __init__(self):
        self.caption = None
        self.word_count = {}
        self.word2index = {}
        self.index2word = {}
        self.num_words  = 4
        self.json_dict  = {}
        self.MAX_LEN    = 37
        self.MIN_COUNT  = 0
        self.json_path  = os.path.abspath(r"data\vocab.json")

        print(self.json_dict, self.json_dict is {})
        if not self.json_dict:
            try:
                with open(self.json_path, 'r') as f:
                    self.word2index = json.load(f)
                    self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))
            except:
                pass
        else:
            self.json_dict['<PAD>']   = PAD_token
            self.json_dict['<START>'] = SOS_token
            self.json_dict['<END>']   = EOS_token
            self.json_dict['<UNK>']   = UNK_token

    def set_min_count(self, min_count):
        """

        """
        self.MIN_COUNT = min_count

    def create_vocab(self, caption, add_to_vocab=False):
        if add_to_vocab:
            self.caption = [self.add_token(token.text.lower()) for token in sp.tokenizer(caption) if not token.is_punct]
        else:
            self.caption = [token.text.lower() for token in sp.tokenizer(caption) if not token.is_punct]

        if len(self.caption) > self.MAX_LEN:
            self.MAX_LEN = len(self.caption)

        return self.caption

    def add_token(self, token):
        if token not in self.word2index:
            self.word2index[token] = self.num_words
            self.index2word[self.num_words] = token
            self.word_count[token] = 1
            self.num_words += 1
        else:
            self.word_count[token] += 1

        if self.word_count[token] >= self.MIN_COUNT:
            self.json_dict[token] = self.word2index[token]

        return token
    
    def store_json(self):
        with open(self.json_path, "w") as outfile:
            json.dump(self.json_dict, outfile, indent=4)

    def get_word_embedding(self, caption):
        caption = [token.text.lower() for token in sp.tokenizer(caption) if not token.is_punct]
        final_caption = []
        final_caption.append(self.word2index['<START>'])
        for token in caption:
            if token not in self.word2index:
                final_caption.append(self.word2index['<UNK>'])
            else:
                final_caption.append(self.word2index[token])

        diff = self.MAX_LEN - len(final_caption)

        if diff!=0:
            final_caption.extend([self.word2index['<PAD>'] for i in range(diff-1)])

        final_caption.append(self.word2index['<END>'])

        return final_caption

    def get_word_token(self, word_embedding):
        return self.index2word[word_embedding]

def convertFromTensor(imageTensor):
    x = imageTensor.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)

    return x

# Display images
def show_imgs(img):
    img  = convertFromTensor(img)
    plt.imshow(img)
    plt.show()