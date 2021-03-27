import torch
from torch.nn import Module, Linear, LSTM, Embedding, Dropout, ReLU
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
        elif backbone == 'resnet50':
            self.base_model = self.resnet50()                 
        elif backbone == 'inception-v3':
            self.base_model = self.inception()
        # elif backbone == 'EfficientNet':
        #     pass
        else:
            print('Please choose a valid backbone.')
            exit()
        
        self.relu = ReLU()
        self.dropout = Dropout(0.5)

    # Backbone Creation: Load weights, freeze layers, replace classifier heads with dense layers
    def vgg16(self):
        model =  models.vgg16(pretrained=True)
        if self.freeze_layers:
            for parameter in model.parameters():
                parameter.requires_grad = False

        model.classifier = Linear(model.classifier[0].in_features, self.embedding_size)
        return model

    def resnet18(self):
        model = models.resnet18(pretrained=True)
        if self.freeze_layers:
            for parameter in model.parameters():
                parameter.requires_grad = False
        
        model.fc = Linear(model.fc.in_features, self.embedding_size)
        return model
    
    def resnet50(self):
        model = models.resnet50(pretrained=True)
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
        x = self.dropout(self.relu(x))

        return x
class Decoder(Module):

    def __init__(self, vocab_size=10000, embedding_size=256, hidden_size=256, lstm_cells=1, lstm_dropout=0.):
        super(Decoder, self).__init__()

        self.vocab_size  = vocab_size
        self.embed_size  = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = lstm_cells

        self.embed  = Embedding(self.vocab_size, self.embed_size)
        self.lstm   = LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=lstm_cells, batch_first=True, dropout=lstm_dropout)
        self.fc1    = Linear(self.hidden_size, self.vocab_size)
        # self.hidden_state = Linear(self.embed_size, self.hidden_size)
        # self.cell_state = Linear(self.embed_size, self.hidden_size)
        
        # self.relu = ReLU()
        self.dropout = Dropout(0.5)
    
    def init_hidden(self, features):
        # hidden = self.relu(self.hidden_state(features)).unsqueeze(0)
        # hidden = torch.cat((hidden, )*self.num_layers, dim=0)
        # cell = self.relu(self.cell_state(features)).unsqueeze(0)
        # cell = torch.cat((cell, )*self.num_layers, dim=0)
        # return (hidden, cell)
        return None

    def forward(self, features, caption):
        hidden = self.init_hidden(features)
        embeddings = self.dropout(self.embed(caption))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(embeddings)
        outputs = self.fc1(outputs)

        return outputs
    
    def predict(self, features, length):
        # Find caption word by word using encoded features and greedy aproach
        hidden = self.init_hidden(features)
        predicted_tokens = []
        for i in range(length):
            # features = (batch_size, 1, embedding_size)        
            outputs, hidden = self.lstm(features, hidden)                           # outputs = (batch_size, 1, hidden_size)
            outputs = self.fc1(outputs.squeeze(1))                                  # outputs = (batch_size, vocab_size)
            predicted_token_i = outputs.max(1)[1]                                   # predicted_token_i = (batchsize)
            predicted_tokens.append(predicted_token_i)
            features = self.embed(predicted_token_i).unsqueeze(1)                   # features = (batch_size, 1, embedding_size)
        return torch.stack(predicted_tokens, 1)

    # # Find caption using beam search for better performance
    # def beam_predict(self, features, max_length, beam_size=3):
    #     batch_size = features.shape(0)

    #     # Features = (batch_size, 1, embed_size)
    #     features = torch.stack([features] * beam_size, dim=1)                   # features = (batch_size, beam_size, 1, embedding_size)
    #     _, hidden = self.lstm(features)
        
    #     # Initialize sequences with <START> values
    #     sequences = torch.ones(batch_size, beam_size, 1, dtype=torch.long)          # sequences = (batch_size, beam_size, 1)
    #     current = sequences[:]                                                      # current   = (batch_size, beam_size, 1)

    #     # Logarithmic score for length normalization
    #     scores = torch.zeros(batch_size, beam_size, 1)                              # score = (batchsize, beam_size, 1)

    #     for i in range(1, max_length):
    #         embeds = self.embed(current)                                            # embeds = (batch_size, beam_size, 1, embedding_size)
    #         outputs, hidden = self.lstm(embeds, hidden)                             # outputs = (batch_size, beam_size, 1, hidden_size)
    #         outputs = torch.log(self.fc1(outputs.squeeze(2)))                       # outputs = (batch_size, beam_size, vocab_size)
    #         preds = outputs + scores                                                # predicted = (batch_size, beam_size, vocab_size)
    #         token_beam = torch.argsort(preds, descending=True)[:, :, :beam_size]    # (batch_size, beam_size, beam_size)
    #         # Store tokens and scores together
    #         current = token_beam.view(batch_size, -1)[:, :beam_size]                # (batch_size, beam_size, 1)
    #         sequences = torch.cat((sequences, current), dim=2)                      # (batch_size, beam_size, i + 1)
    #         scores = torch.gather(outputs)



class Model(Module):
    def __init__(self, backbone, freeze_layers, embed_size, hidden_size, vocab_size, lstm_cells, lstm_dropout, verbose, device):
        super(Model, self).__init__()

        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.num_rnn     = lstm_cells
        self.dropout     = lstm_dropout
        self.device      = device

        self.encoder = Encoder(backbone=backbone, embedding_size=self.embed_size, freeze_layers=freeze_layers).to(self.device)
        self.decoder = Decoder(embedding_size=self.embed_size, 
                            hidden_size=self.hidden_size, 
                            vocab_size=self.vocab_size, 
                            lstm_cells=self.num_rnn, 
                            lstm_dropout=self.dropout).to(self.device)

        if verbose:
            print("*"*20)
            print("Encoder Architecture : ")
            print("*"*20)
            summary(self.encoder.cuda(), shape=(1, 224, 224))

            print()

            print("*"*20)
            print("Decoder Architecture : ")
            print("*"*20)
            summary(self.encoder.cuda(), shape=(1, 224, 224))
    
    def forward(self, data):
        image, caption = data['image'].to(self.device), data['caption'].to(self.device)

        img_features      = self.encoder(image)
        caption_predicted = self.decoder(img_features, caption[:, :-1])

        return caption_predicted
    
    # Predicts captions using given images
    def predict(self, images, caption_length, beam=None):
        with torch.no_grad():
            # Encode images = (batch_size, 3, 224, 224) to features = (batch_size, self.embedding_size)
            features = self.encoder(images.to(self.device))
            if beam is None:
                predicted_tokens = self.decoder.predict(features.unsqueeze(1), caption_length)
            # else:
            #     predicted_tokens = self.decoder.beam_predict(features.unsqueeze(1), caption_length, beam)
            return predicted_tokens

if __name__ == '__main__':
    """ d = Decoder(8, 12, 16, 1)
    output = d(torch.ones(1, 12), torch.ones(1, 39).long())
    print(output.shape) """

    # d = Decoder(20, 10, 12, 1)
    # print(d.fc1.weight[0])
    
    # loss = BCELoss()
    
    # o = torch.optim.Adam(d.parameters(), lr=0.001)
    # for i in range(100):
    #     features = torch.ones(1, 10)
    #     captions = torch.ones(1, 39)
    #     output, hidden = d(features, captions.long())
    #     l = loss(torch.nn.functional.softmax(output), torch.ones_like(output)*1000)
    #     o.zero_grad()
    #     l.backward()
    #     o.step()
    # print(d.fc1.weight[0])
    # print(d)