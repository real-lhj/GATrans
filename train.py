# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
from datetime import datetime
import torch.nn as nn

# from args import get_parser
from utils import *

# from prediction import Predictor
from training import Trainer
from gatrans import GATrans

if __name__ == '__main__':

    # 1. args
    batch_size = 256
    init_lr = 1e-2
    #
    x_train = [np.random.rand(8) for i in range(5000)]
    x_test = [np.random.rand(8) for i in range(500)]
    window_size = 10
    n_features = target_dims = 8
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(  # 获取小批量数据
        train_dataset, batch_size, val_split=0.0, shuffle=False, test_dataset=None
    )

    out_dim = n_features
    # 2. build model
    model = GATrans(
        n_features,
        window_size,
        out_dim,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,

        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    )
    # 3.optimizer and lossfunction
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    # 4.start to train
    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=50,
        batch_size=16,
        init_lr=0.0001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
    )

    trainer.fit(train_loader, val_loader)



