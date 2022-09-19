"""
使用transformer进行时间序列预测
"""
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=38, num_layers=3, dropout=0.1):
        super(TransAm, self).__init__()
        self.transformer = nn.Transformer(d_model=feature_size, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=512,
                                          batch_first=True)
        self.pe = PositionalEncoding(d_model=feature_size, dropout=0)

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src = src * math.sqrt(self.n_feats)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

torch.manual_seed(0)
np.random.seed(0)
input_window = 100
output_window = 5
batch_size = 32 # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
