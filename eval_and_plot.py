import json
from datetime import datetime

import torch
import torch.nn as nn

# from args import get_parser
from utils import *
from sklearn.preprocessing import MinMaxScaler, RobustScaler
# from prediction import Predictor
from training import Trainer
from gatrans import GATrans
from lstm import LSTM
from encoder_decoder import Encoder_Decoder
from transformer_encoder_decoder import TransEncDec
import  pandas as pd
import  os
from tqdm import tqdm
from predicting import Predictor
torch.manual_seed(3407)   #为了使用同样的随机初始化种子以形成相同的随机效果

if __name__ == '__main__':

    # 1. args
    save_model_path = "./saved_model/SMD"
    batch_size = 128
    init_lr = 1e-3
    epoch = 30
    dataset = "SMD"
    group = "3-1"
    group_index = group[0]
    index = group[2]
    window_size = 50
    n_features = 38
    output_window = 10
    dropout = 0.
    use_cuda = True
    model_name = "trans"
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    (x_train, _), (x_test, y_test), scaler = get_data(f"machine-{group_index}-{index}", normalize=True)
    train_dataset = ShiftWindowDataset(x_train, window_size, n_features, horizon=output_window)
    test_dataset = ShiftWindowDataset(x_test, window_size, n_features, horizon=output_window)

    train_loader, val_loader = create_data_loaders(  # 获取小批量数据
        train_dataset, batch_size, val_split=0.0, shuffle=False
    )


    # 2. build model
    model_trans = TransEncDec(window_size, output_window, n_features)
    model_name = "transformer_SMD_3-1_project.pt"
    model_trans.load_state_dict(torch.load(os.path.join(save_model_path, model_name), map_location=device))
    model_trans.to(device)

    # model2 = Encoder_Decoder(window_size+1, n_features, 128, 64)
    # model_name = "encoder_decoder_SMD1-2.pt"
    # model2.load_state_dict(torch.load(os.path.join(save_model_path, model_name), map_location=device))
    # model2.to(device)
    #
    # model_lstm = LSTM(input_size=38, hidden_size=128, num_layers=1)
    # model_mame = "lstm_SMD3-1.pt"
    # model_lstm.load_state_dict(torch.load(os.path.join(save_model_path, model_mame), map_location=device))
    # model_lstm.to(device)

    # Some suggestions for POT args

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    plot(model_trans, test_loader, n_features, scaler, save_path="./prediction_figure/SMD_3-1")