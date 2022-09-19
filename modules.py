import torch
import torch.nn as nn
import math
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout) #表示每个神经元有p的可能性不被激活

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #shape (mat_len, 1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        # pe[:, 0::2] = torch.sin(position * div_term) #奇数列
        # pe[:, 1::2] = torch.cos(position * div_term) #偶数列
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :] #x(10,128,76)
        # return self.dropout(x) #x(10,128,76)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model) #d_model = feats*2 = 38*2 = 76
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2) # 应该代表残差连接
        return src # encoder的输出(10,128,76)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        # d_model = embed_dim
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0] # memory是encoder的输出.
        # tgt和memory的维度可以不一样
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt






