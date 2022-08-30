import torch
import torch.nn as nn

class FeatureAttentionLayer(nn.Module):
    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.num_nodes = n_features
        self.embed_dim *= 2
        lin_input_dim = 2 * window_size
        a_input_dim = self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.bias = nn.Parameter(torch.empty(n_features, n_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape[b, n, k]
        x = x.permute(0, 2, 1) #[b, k, n]
        a_input = self.__make_attention_input(x)
        tmpret = self.lin(a_input)
        a_input = self.leakyrelu(tmpret)
        e = torch.matmul(a_input, self.a).squeeze(3)
        e += self.bias

        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def __make_attention_input(self, v):
        K = self.num_nodes
        left = v.repeat_interleave(K, dim=1) #(b, K*K, n)
        right = v.repeat(1, K, 1) #(b, K*K, n)
        combined = torch.cat((left, right), dim=2) #(b, K*K, 2*n)

        return combined.view(v.size(0), K, K, 2*self.window_size)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.num_nodes = window_size

        self.embed_dim *= 2
        lin_input_dim = 2 * n_features
        a_input_dim = self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.bias = nn.Parameter(torch.empty(window_size, window_size))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape[b, n, k]

        a_input = self.__make_attention_input(x)
        a_input = self.leakyrelu(self.lin(a_input))
        e = torch.matmul(a_input, self.a).squeeze(3)
        e += self.bias

        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h = self.sigmoid(torch.matmul(attention, x))
        return h

    def __make_attention_input(self, v):
        K = self.num_nodes
        left = v.repeat_interleave(K, dim=1)  # (b, n*n, k)
        right = v.repeat(1, K, 1)  # (b, n*n, K)
        combined = torch.cat((left, right), dim=2)  # (b, n*n, 2*k)

        return combined.view(v.size(0), K, K, 2 * self.n_features)

class GRULayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        # x shape[b, n, 3k]
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]
        return out, h


class ForecastingModel(nn.Module):
    # 预测模型：多个全连接层堆叠
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ForecastingModel, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # print(self.lin)
        # return self.lin(x)
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)

class RNNDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out #[b, 10, 150]


class ReconstructionModel(nn.Module):
    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim) #[b, 10, 150]=>[b, 10, 38]

    def forward(self, x):
        h_end = x #(b, 150)
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1)
        h_end_rep = h_end_rep.view(x.size(0), self.window_size, -1) #(b, 10, 150)==(b,seq_len,embed_dim)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out








