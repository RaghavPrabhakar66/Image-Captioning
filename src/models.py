import torch
import torchvision
from torchsummary import summary
#from pytorch_model_summary import summary
import torch.nn as nn
from torchvision.models import resnet18, vgg16

class Encoder(nn.Module):

    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.base_model = None
        self.embedding_size = embedding_size

    def vgg16_backbone(self):
        self.base_model = vgg16(pretrained=True)
        
        """# save the num of input features for the current fc layer
        input_features_last_layer = self.classifier.fc.in_features

        # freeze all the layers
        for p in self.base_model.parameters():
            p.requires_grad = False

        #replace the fc layer with our own
        self.base_model.classifier[-1] = nn.Linear(in_features=input_features_last_layer, out_features=self.embedding_size)
        
        # make the last layers params trainable
        #for p in self.base_model.fc.parameters():
        #    p.requires_grad = True"""

    def forward(self, image):
        x = self.base_model(image)
        output = torch.sigmoid(x)

        return output

class Decoder(nn.Module):

    def __init__(self, vocab_size=10000, embedding_size=256, hidden_size=256, rnn_cells=128, rnn_dropout=0.):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embedding_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=rnn_cells, batch_first=True, dropout=rnn_dropout)
        self.fc1 = nn.Linear(self.hidden_size, self.vocab_size)
        # self.fc2 = nn.Linear(rnn_cells, self.vocab_size)

    def init_hidden_state(self):
        x = torch.zeros(self.hidden_size)

    def forward(self, img_features, caption):
        """
        img_features => [4, 128] -> [batch, embed]
        caption.shape => [36, 4] -> [cap_len-1, batch]
        embeddings.shape => [36, 4, 128]
        img_features.unsqueeze(0).shape => [1, 4, 128]
        embeddings.shape => [4, 37, 128]
        The index error with embeddings was due to the fact that the original
        vocab was created based on 10000 as max, but then the final vocab size
        is 2983, which is less than 10000. There are some words in the vocab,
        which have a index higher the len(vocab), so if we specify our
        nn.Embedding to be of len(vocab), words with index higher than that
        will throw errors.
        Soln: set the embedding layer to be the max index size in the vocab
        https://stackoverflow.com/questions/56010551/pytorch-embedding-index-out-of-range
        """

        # Permute makes sure that concat with img_features work correctly
        embeddings = self.embed(caption)

        embeddings = torch.cat((img_features.unsqueeze(0), embeddings), dim=0)

        output, _ = self.rnn(embeddings)
        outputs = self.fc1(output)

        return outputs

class Model(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, rnn_units, dropout):
        super(Model, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_rnn = rnn_units
        self.dropout = dropout

        self.encoder = Encoder(embedding_size=self.embed_size)
        self.decoder = Decoder(embedding_size=self.embed_size, 
                            hidden_size=self.hidden_size, 
                            vocab_size=self.vocab_size, 
                            rnn_cells=self.num_rnn, 
                            rnn_dropout=self.dropout)

if __name__ == '__main__':
    encoder = Encoder(128)
    print(encoder)
    summary(encoder.cuda(), input_size=(1, 3, 224, 224), verbose=1, depth=3)
    #print(summary(Encoder(128), torch.zeros((1, 1, 28, 28)), show_input=True, show_hierarchical=True))
