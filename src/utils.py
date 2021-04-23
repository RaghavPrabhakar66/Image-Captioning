from collections import Counter
from string import punctuation
import os
import json
import matplotlib.pyplot as plt
import discord
from discord import Webhook, RequestsWebhookAdapter

PAD_token = 0   # Used for padding short sentences
SOS_token = 1   # Start-of-sentence token
EOS_token = 2   # End-of-sentence token
UNK_token = 3   # Unkown word token

class VocabBuilder:
    def __init__(self, min_count=1, json_path=None):
        self.MIN_COUNT  = min_count
        self.json_path  = os.path.abspath(json_path)

    def generate_vocab(self, words, maxlen):
        count = Counter(words)
        sorted_count = count.most_common(len(words))

        json_list = [('<PAD>', PAD_token), ('<START>', SOS_token), ('<END>', EOS_token), ('<UNK>', UNK_token)]
        json_list.extend([(word, index + 4) for index, (word, freq) in enumerate(sorted_count) if freq >= self.MIN_COUNT])
        json_dict = dict(json_list)

        with open(self.json_path, "w") as outfile:
            json.dump({'vocab_dict': json_dict, 'maxlen': maxlen+2}, outfile, indent=4)

class Vocab:
    def __init__(self, file):
        self.word2index = {}
        self.index2word = {}
        self.MAX_LEN    = 0
        self.MAX_INDEX  = 0
        self.vocab_path  = os.path.abspath(file)

        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r') as f:
                vocab_data = json.load(f)
                self.word2index = vocab_data['vocab_dict']
                self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))
                self.MAX_LEN = vocab_data['maxlen']
                self.MAX_INDEX = max(self.word2index.values()) + 1
        else:
            print('No file provided to vocab')

    def get_word_embedding(self, caption):
        # caption = [token.text.lower() for token in sp.tokenizer(caption) if not token.is_punct]
        caption = caption.translate({ord(i): '' for i in punctuation}).lower().split()
        
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
        try:
            assert len(final_caption) == self.MAX_LEN
        except:
            print('\nLength was not correct:', len(final_caption))
            raise AssertionError

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
    
