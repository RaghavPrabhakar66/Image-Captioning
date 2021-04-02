from collections import Counter
import os
import spacy
import json
import matplotlib.pyplot as plt
import discord
from discord import Webhook, RequestsWebhookAdapter

sp = spacy.load("en_core_web_sm")  # load english core

PAD_token = 0   # Used for padding short sentences
SOS_token = 1   # Start-of-sentence token
EOS_token = 2   # End-of-sentence token
UNK_token = 3   # Unkown word token

class VocabBuilder:
    def __init__(self, min_count=None, json_path=None):
        self.word_count = {}
        self.word2index = {}
        self.num_words  = 4
        self.json_dict  = {}
        self.MAX_LEN    = 37
        self.MIN_COUNT  = min_count
        self.json_path  = os.path.abspath(json_path)

    def add_to_vocab(self, captions):
        count = Counter(captions)
        sorted_count = count.most_common(len(captions))

        json_list = [('<PAD>', PAD_token), ('<START>', SOS_token), ('<END>', EOS_token), ('<UNK>', UNK_token)]
        json_list.extend([(word, index + 4) for index, (word, freq) in enumerate(sorted_count) if freq >= self.MIN_COUNT])
        self.json_dict = dict(json_list)

    def store_json(self):
        with open(self.json_path, "w") as outfile:
            json.dump(self.json_dict, outfile, indent=4)

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.json_dict  = {}
        self.MAX_LEN    = 37 + 2
        self.MAX_INDEX  = 0
        self.vocab_path  = os.path.abspath("data/vocab.json")

        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r') as f:
                self.word2index = json.load(f)
                self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))
                self.MAX_INDEX = max(self.word2index.values()) + 1
        else:
            print('File does not exist.')

    def get_word_embedding(self, caption):
        caption = [token.text.lower() for token in sp.tokenizer(caption) if not token.is_punct]
        
        final_caption = []
        final_caption.append(self.word2index['<START>'])

        for token in caption:
            if token not in self.word2index:
                final_caption.append(self.word2index['<UNK>'])
            else:
                final_caption.append(self.word2index[token])
        final_caption.append(self.word2index['<END>'])

        diff = self.MAX_LEN - len(final_caption)

        if diff > 0:
            final_caption.extend([self.word2index['<PAD>'] for i in range(diff)])
        assert len(final_caption) == self.MAX_LEN
        return final_caption

    def get_word_token(self, word_embedding):
        return self.index2word[word_embedding]
    
    def __len__(self):
        return len(self.word2index)

def convertFromTensor(imageTensor):
    x = imageTensor.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)

    return x

# Display images
def show_imgs(img):
    img  = convertFromTensor(img)
    plt.imshow(img)
    plt.show()
    
class DiscordNotifier:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url
    
    def send_message(self, training_loss, val_loss, epoch, total_epochs, name, save_path):
        webhook = Webhook.from_url(self.webhook_url, adapter=RequestsWebhookAdapter())
        
        title  = "Epoch: " + str(epoch) + " of " + str(total_epochs)
        footer = "saved model: " + save_path

        embed=discord.Embed(title=title, description=" ", color=0x8a0085)
        embed.set_author(name=name)
        embed.set_thumbnail(url="https://pytorch.org/assets/images/pytorch-logo.png")
        embed.add_field(name="Training Loss", value=training_loss, inline=False)
        embed.add_field(name="Validation Loss", value=val_loss, inline=False)
        embed.set_footer(text=footer)
    
        webhook.send(embed=embed)
    
