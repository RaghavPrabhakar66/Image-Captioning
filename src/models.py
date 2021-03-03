import torch as T
from torch.nn import Module, Linear, LSTM
import torch.functional as F
from torch.nn.modules import module

class Encoder(Module):
    def __init__(self):
        super(self, ).__init__()
        self.