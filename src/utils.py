import os
import spacy
import json
import matplotlib.pyplot as plt
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

        self.json_dict['<PAD>']   = PAD_token
        self.json_dict['<START>'] = SOS_token
        self.json_dict['<END>']   = EOS_token
        self.json_dict['<UNK>']   = UNK_token

    def add_to_vocab(self, caption):
        caption = [self.add_token(token.text.lower()) for token in sp.tokenizer(caption) if not token.is_punct]

        if len(caption) > self.MAX_LEN:
            self.MAX_LEN = len(self.caption)

        return caption

    def add_token(self, token):
        if token not in self.word2index:
            self.word2index[token] = self.num_words
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

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.json_dict  = {}
        self.MAX_LEN    = 37
        self.MAX_INDEX  = 0
        self.vocab_path  = os.path.abspath("data/vocab.json")

        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r') as f:
                self.word2index = json.load(f)
                self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))
                self.MAX_INDEX = max(self.word2index.values())
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

        diff = self.MAX_LEN - len(final_caption)

        if diff!=0:
            final_caption.extend([self.word2index['<PAD>'] for i in range(diff-1)])

        final_caption.append(self.word2index['<END>'])

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
    def __init__(self, webhook_url=None)
        self.webhook_url = webhook_url
    
    def notify_discord(self, training_loss, training_acc, val_loss, val_acc, epoch, total_epochs, name, save_path):
        webhook = Webhook.from_url(self.webhook_url, adapter=RequestsWebhookAdapter())
        
        title  = "Epoch: " + str(epoch) + " of " + str(total_epochs)
        footer = "saved model: " + save_path

        embed=discord.Embed(title=title, description=" ", color=0x8a0085)
        embed.set_author(name=name)
        embed.set_thumbnail(url="https://pytorch.org/assets/images/pytorch-logo.png")
        embed.add_field(name="Training Accuracy", value=training_acc, inline=False)
        embed.add_field(name="Training Loss", value=training_loss, inline=False)
        embed.add_field(name="Validation Accuracy", value=val_acc, inline=False)
        embed.add_field(name="Validation Loss", value=val_loss, inline=False)
        embed.set_footer(text=footer)
    
        webhook.send(embed=embed)
    
