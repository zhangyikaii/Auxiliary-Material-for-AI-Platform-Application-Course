import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def basic_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Sigmoid()
    )

class Autoencoder(nn.Module):
    def __init__(self, layers=[784, 2000, 1000, 500, 30]):
        super(Autoencoder, self).__init__()
        self.encoder = "TODO"
        self.decoder = "TODO"

    def forward(self, x):
        hidden = "TODO"
        reconstructed = self.decoder(hidden)
        return reconstructed
