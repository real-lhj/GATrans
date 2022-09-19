import json
from datetime import datetime

import torch
import torch.nn as nn
from utils import *
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
    save_model_path = "./saved_model/PSM"
    batch_size = 256
    init_lr = 1e-3
    # epoch = 30
    dataset = "PSM"
    # group = "1-2"
    # group_index = group[0]
    # index = group[2]
    window_size = 100
    output_window = 30
    n_features = 38
    dropout = 0.
    use_cuda = True
    model_name = "transformer"
    test_name = "test_20001_40001.csv"
    label_name = "test_label_20001_40001.csv"




    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    (x_train, _), (x_test, y_test), scaler = get_splited_data(dataset, test_name, label_name, normalize=True)
    n_features = x_test.shape[1]
    train_dataset = ShiftWindowDataset(x_train, window_size, n_features, horizon=output_window)
    test_dataset = ShiftWindowDataset(x_test, window_size, n_features, horizon=output_window)
    # train_loader, val_loader = create_data_loaders(  # 获取小批量数据
    #     train_dataset, batch_size, val_split=0.0, shuffle=False,
    # )
    # 2. build model
    model_trans = TransEncDec(window_size, output_window, n_features)
    model_name = "transformer_PSM_nhead25.pt"
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
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
    }
    # key = "SMD-" + group[0]
    key = "SMD-1"
    level, q = level_q_dict[key]

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    # key = "SMD-" + group[0]
    key = "SMAP"
    reg_level = reg_level_dict[key]

    prediction_args = {
        'dataset': dataset,
        "target_dims": n_features,
        'scale_scores': False,
        "level": level,
        "q": q,
        'dynamic_pot': False,
        "use_mov_av": False,
        "gamma": 1,
        "reg_level": reg_level,
        "save_path": f"{save_model_path}",
        "scaler": scaler
    }

    label = y_test[window_size - 1:] # 第一个元素是tgt[:, -1, :]
    predictor = Predictor(model_trans, window_size, output_window, n_features, prediction_args)
    predictor.predict_anomalies(x_train, x_test, label)
