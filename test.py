import  torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
# A = torch.Tensor(5,2,4) #(seq,bs,emb)
# nn.init.xavier_normal_(A)
# print(A)
# M = nn.MultiheadAttention(embed_dim=4, num_heads=2)
# attention_mask = ~torch.tril(torch.ones([A.shape[0],A.shape[0]])).bool() #某个位置true表示掩盖该位置
# print(attention_mask)
# attn_output, attn_output_weights=M(A,A,A, attn_mask=attention_mask)
# print(attn_output)
# print(attn_output.shape)
# print(attn_output_weights)

# x = torch.tensor([[1.,1.,1.],
#             [2.,3.,4.]], dtype=torch.float32)
# y = torch.tensor([[2.,2.,4.],
#             [10.,10.,10.]], dtype=torch.float32)
# loss = nn.MSELoss()
# l = loss(x, y)
# print(l)

# path = "./data/ServerMachineDataset/test" #文件夹目录
# files= os.listdir(path) #得到文件夹下的所有文件名称
# lenth = 0
# for file in files:
#     data = pd.read_csv(os.path.join(path, file))
#     lenth += len(data)
# print(lenth)

# a = torch.from_numpy(np.random.randint(1, 5, size=(3, 10)))
# print(a)
# print(a.shape)
# x = np.arange(0, 24)
# x = x.reshape((2, 3, 4)) #(b,k,n)
# x = torch.from_numpy(x).float()
# print(x)
# print(f'x.shape:{x.shape}')
# b = x.repeat(1,3,1)
# print(f'b.shape:{b.shape}')
# # print(b)
# print('=' * 50)
# c = x.repeat_interleave(3, dim=1)
# print(f'c.shape:{c.shape}')
# # print(c)
# combind = torch.cat((c, b), dim=2)
# print(combind.shape)
# print(combind)
# a_input = combind.view(x.size(0), 3, 3, 2*4)
# print(f'a_input.shape:{a_input.shape}')
# print(a_input)
#
# print('='*20 + 'the gatv2 process' + '='*20)
# lin = nn.Linear(8, 8)
# leakyrelu = nn.LeakyReLU(0.2)
# sigmoid = nn.Sigmoid()
# a = nn.Parameter(torch.empty((8, 1)))
# nn.init.xavier_uniform_(a.data, gain=1.414)
# bias = nn.Parameter(torch.empty(3, 3))
#
# tmpret = lin(a_input) #(2,3,3,8)
# a_input = leakyrelu(tmpret) #(2,3,3,8)
# e = torch.matmul(a_input, a).squeeze(3) #(2,3,3)
# e += bias
# attention = torch.softmax(e, dim=2) #(2,3,3)
# attention = torch.dropout(attention, 0.2, train=True) #(2,3,3)
# h = sigmoid(torch.matmul(attention, x))
# print(f"h.shape{h.shape}") #(2,3,4)
# print(h)
#
# fc = nn.Linear(4, 5)
# out = fc(h)
# print(f"out.shape={out.shape}")
from sklearn import preprocessing
import numpy as np
X = np.array([[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]])
scaler= preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X)
X_scaled = scaler.transform(X)
print(X)
print(X_scaled)
X1 = scaler.inverse_transform(X_scaled)
print(X1)
print(X1[0, -1])





