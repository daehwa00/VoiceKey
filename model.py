from modules import conv as m
import torch
from torch import nn


class EVACModel(nn.Module):
    def __init__(self, cnn: m.Conv1DNet = m.Conv1DNet()):
        super().__init__()
        self.cnn = cnn

    def forward(self, x):
        return self.cnn(x)
