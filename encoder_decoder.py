import torch
import torch.nn as nn
class Encoder_Decoder(nn.Module):# 编码器-解码器生成模型
    def __init__(self, window_size, n_feats, hidden1, hidden2):
        super(Encoder_Decoder, self).__init__()
        self.model_name = "encoder_decoder"
        self.input_size = window_size * n_feats
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden1), nn.ReLU(True),
            nn.Linear(self.hidden1, 32), nn.ReLU(True),
            nn.Linear(32, self.hidden2), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden2, self.hidden1), nn.ReLU(True),
            nn.Linear(self.hidden1, 32), nn.ReLU(True),
            nn.Linear(32, self.input_size),
            nn.Sigmoid()
        )
    def forward(self, x, y):
        # x.shape [b, n, k]
        # y.shape [b, 1, k]
        input = torch.cat((x, y), dim=1)
        n = input.shape[1]
        k = input.shape[2]
        input = input.view(-1, n*k) #(b, (n+1)*k)
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded.reshape(-1, n, k)






