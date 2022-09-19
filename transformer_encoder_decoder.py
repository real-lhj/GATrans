import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
class PositionalEncoding(nn.Module):
  "Implement the PE function."
  def __init__(self, d_model, dropout, max_len=5000):
    #d_model=512,dropout=0.1,
    #max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
    #一般100或者200足够了。
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    #(5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
    #每个位置用一个512维度向量来表示其位置编码
    position = torch.arange(0, max_len).unsqueeze(1)
    # (5000) -> (5000,1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
      -(math.log(10000.0) / d_model))
      # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1]) # 奇数下标的位置
    else:
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
    pe = pe.unsqueeze(0)
    # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
    self.register_buffer('pe', pe)
  def forward(self, x):
    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    # 接受1.Embeddings的词嵌入结果x，
    #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
    #例如，假设x是(30,10,512)的一个tensor，
    #30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
    #则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
    #在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
    #保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
    return self.dropout(x) # 增加一次dropout操作
# 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数。

class TransEncDec(nn.Module):
    def __init__(self, window_size, horizon, n_feats):
        super(TransEncDec, self).__init__()
        self.model_name = "transformer"
        self.n_feats = n_feats # 25 for PSM
        self.window = window_size
        self.output_window = horizon
        self.dropout = 0.
        self.num_layers = 3
        self.projected_dim = 512
        self.nhead = 8  # d_model must be divided by nhead, 19 for SMD; 5 for PSM
        self.src_pos_encoder = PositionalEncoding(self.projected_dim, self.dropout)
        self.tgt_pos_encoder = PositionalEncoding(self.projected_dim, self.dropout)
        self.transformer = nn.Transformer(d_model=self.projected_dim, nhead=self.nhead, num_encoder_layers=self.num_layers,
                                          num_decoder_layers=self.num_layers, dim_feedforward=64,
                                          batch_first=True, device="cuda")
        self.src_mask = None
        self.encoder_projection_layer = nn.Linear(self.n_feats, self.projected_dim) # 把feats投影到512
        self.decoder_projection_layer = nn.Linear(self.projected_dim, self.n_feats) # 再把512投影回n_feats输出

    def forward(self, src, tgt):
        # x[b, n, k]
        src = self.encoder_projection_layer(src)
        src = src * math.sqrt(self.projected_dim)
        src = self.src_pos_encoder(src)
        tgt = self.encoder_projection_layer(tgt)
        tgt = tgt * math.sqrt(self.projected_dim)
        tgt = self.tgt_pos_encoder(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-2])  # n
        tgt_mask = tgt_mask.to("cuda")
        tgt_y = self.transformer(src, tgt, tgt_mask=tgt_mask)

        return self.decoder_projection_layer(tgt_y)

        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        # # print('a',src.size())
        # mask = nn.Transformer.generate_square_subsequent_mask(src.size()[-2]).to(device)
        # self.src_mask = mask
        #
        # src = self.pos_encoder(src)
        # # print('j',src.size(),self.src_mask.size())
        # output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask) # 为什么在encoder里加下三角mask呢
        #
        # output = self.decoder(output)
        #
        # return output


class TransEncDec_withoutProjection(nn.Module):
    def __init__(self, window_size, horizon, n_feats):
        super(TransEncDec_withoutProjection, self).__init__()
        self.model_name = "transformer"
        self.n_feats = n_feats # 25 for PSM
        self.window = window_size
        self.output_window = horizon
        self.dropout = 0.
        self.num_layers = 3
        # self.projected_dim = 512
        self.nhead = 38  # d_model must be divided by nhead, 38 for SMD; 5 for PSM
        self.src_pos_encoder = PositionalEncoding(n_feats, self.dropout)
        self.tgt_pos_encoder = PositionalEncoding(n_feats, self.dropout)
        self.transformer = nn.Transformer(d_model=n_feats, nhead=self.nhead, num_encoder_layers=self.num_layers,
                                          num_decoder_layers=self.num_layers, dim_feedforward=64,
                                          batch_first=True, device="cuda")

        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(self.n_feats)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_feats, nhead=self.nhead, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.decoder = nn.Linear(self.n_feats, self.n_feats)  # 使用线性层作为输出
        # self.encoder_projection_layer = nn.Linear(self.n_feats, self.projected_dim) # 把feats投影到512
        # self.decoder_projection_layer = nn.Linear(self.projected_dim, self.n_feats) # 再把512投影回n_feats输出

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        # x[b, n, k]
        # src = self.encoder_projection_layer(src)
        src = src * math.sqrt(self.n_feats)
        src = self.src_pos_encoder(src)
        # tgt = self.encoder_projection_layer(tgt)
        tgt = tgt * math.sqrt(self.n_feats)
        tgt = self.tgt_pos_encoder(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-2]) #n
        tgt_mask = tgt_mask.to("cuda")
        tgt_y = self.transformer(src, tgt, tgt_mask=tgt_mask)

        return tgt_y

        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        # # print('a',src.size())
        # mask = nn.Transformer.generate_square_subsequent_mask(src.size()[-2]).to(device)
        # self.src_mask = mask
        #
        # src = self.pos_encoder(src)
        # # print('j',src.size(),self.src_mask.size())
        # output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask) # 为什么在encoder里加下三角mask呢
        #
        # output = self.decoder(output)
        #
        # return output





