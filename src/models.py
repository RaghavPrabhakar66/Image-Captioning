import torch as T
from torch.nn import Module, Linear, LSTM
import torch.nn.functional as F
from torchvision import models

# To-do


# CNN encoder using pretrained models
class Encoder(Module):
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

en = Encoder('vgg16', 10)
print(en(T.zeros((1, 3, 224, 224))))