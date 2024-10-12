import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Skipcon(nn.Module):
    def __init__(self, model_dim):
        super(Skipcon, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.dropout = 0.1
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, input, x):
        residual = x
        out = self.dropout1(input)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out


class con2(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(con2, self).__init__()
        self.in_channels = in_channels
        self.con2out = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size),
                                 # out=2*in 32/128/307/10  out4*in 32/256307/10
                                 stride=(1, time_strides))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        return x_causal_conv


class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))


    def forward(self, x):
        x_gtu = self.con2out(x)
        return x_gtu










