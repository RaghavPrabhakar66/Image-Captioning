import torch
from torch.nn import Module, Linear, LSTM, Embedding
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary


class Encoder(Module):
    """
    CNN encoder using pretrained models
    """
    def __init__(self, backbone, embedding_size, freeze_layers=True):
        super(Encoder, self).__init__()
        
        self.freeze_layers = freeze_layers
        self.embedding_size = embedding_size

        # Choosing the Pretrained model backbone
        if backbone == 'vgg16':
            self.base_model = self.vgg16()
        elif backbone == 'resnet18':
            self.base_model = self.resnet18()
        elif backbone == 'inception-v3':
            self.base_model = self.inception()
        # elif backbone == 'EfficientNet':
        #     pass
        else:
            self.base_model = self.vgg16()

        # Linear layer to produce latent vector
        self.embed = Linear(self.embedding_size, self.embedding_size)

    # Backbone Creation: Load weights, freeze layers, replace classifier heads with dense layers
    def vgg16(self):
        model =  models.vgg16(pretrained=True)
        if self.freeze_layers:
            for parameter in model.parameters():
                parameter.requires_grad = False

        model.classifier = Linear(model.classifier[0].in_features, self.embedding_size)
        return model

    def resnet(self):
        model = models.resnet18(pretrained=True)
        if self.freeze_layers:
            for parameter in model.parameters():
                parameter.requires_grad = False
        
        model.fc = Linear(model.fc.in_features, self.embedding_size)
        return model

    def inception(self):
        model = models.inception_v3(pretrained=True)
        if self.freeze_layers:
            for parameter in model.parameters():
                parameter.requires_grad = False
        
        model.fc = Linear(model.fc.in_features, self.embedding_size)

    # Foward pass
    def forward(self, input):
        x = self.base_model(input)
        x = F.relu(self.embed(x))

        return x
class Decoder(Module):

    def __init__(self, vocab_size=10000, embedding_size=256, hidden_size=256, lstm_cells=128, lstm_dropout=0.):
        super(Decoder, self).__init__()

        self.vocab_size  = vocab_size
        self.embed_size  = embedding_size
        self.hidden_size = hidden_size

        self.embed = Embedding(self.vocab_size, self.embed_size)
        self.lstm   = LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=lstm_cells, batch_first=True, dropout=lstm_dropout)
        self.fc1   = Linear(self.hidden_size, self.vocab_size)

    def init_hidden_state(self):
        x = torch.zeros(self.hidden_size)

        return x

    def forward(self, img_features, caption):
        embeddings = self.embed(caption)
        embeddings = torch.cat((img_features.unsqueeze(1), embeddings), dim=1)
        output, _ = self.lstm(embeddings)
        outputs = self.fc1(output)

        return outputs

class Model(Module):
    def __init__(self, backbone, embed_size, hidden_size, vocab_size, lstm_cells, lstm_dropout, verbose, device):
        super(Model, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_rnn = lstm_cells
        self.dropout = lstm_dropout
        self.device  = device

        self.encoder = Encoder(backbone=backbone, embedding_size=self.embed_size)
        self.decoder = Decoder(embedding_size=self.embed_size, 
                            hidden_size=self.hidden_size, 
                            vocab_size=self.vocab_size, 
                            lstm_cells=self.num_rnn, 
                            lstm_dropout=self.dropout)

        self.encoder.to(device)
        self.decoder.to(device)

        if verbose:
            print("*"*20)
            print("Encoder Architecture : ")
            print("*"*20)
            summary(self.encoder.cuda(), shape=(1, 224, 224))
    
    def forward(self, data):
        image, caption = data['image'].to(self.device), data['caption'].to(self.device)

        img_features      = self.encoder(image)
        caption_predicted = self.decoder(img_features, caption[:, :-1])

        return caption_predicted


if __name__=='__main__':
    en = Encoder('vgg16', 10)
    print(en(torch.zeros((1, 3, 224, 224))))
