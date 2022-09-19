# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import time
from datetime import datetime
# from args import get_parser
import torch
from utils import *
# from prediction import Predictor
from training import Trainer
from gatrans import GATrans
from lstm import LSTM
from encoder_decoder import Encoder_Decoder
from transformer_encoder_decoder import TransEncDec
import  pandas as pd
import  os
from tqdm import tqdm
from args import get_parser
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(3407)   #为了使用同样的随机初始化种子以形成相同的随机效果

def evaluate_transformer(model, data_loader):
    """

    :param model:
    :param data_loader:
    :return: eval_loss
    """
    model.eval()
    eval_loss = 0.
    eval_num = 0
    loss_func = nn.MSELoss()
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to("cuda").to(torch.float32)
            tgt = tgt.to("cuda").to(torch.float32)
            output = model(src, tgt) #(b, n, k)
            loss = loss_func(output[-model.output_window:], tgt[-model.output_window:])
            eval_loss += loss.item() * src.size(0)
            eval_num += src.size(0)
        eval_loss = eval_loss / eval_num
    return eval_loss

if __name__ == '__main__':
    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    # 1. args
    save_model_path = "./saved_model/PSM"
    batch_size = 256
    init_lr = 1e-3
    epoch = 30
    # group = "1-2"
    # group_index = group[0]
    # index = group[2]
    window_size = 50
    output_window = 10
    n_features = None
    dropout = 0.01
    model_name = "transformer_PSM_nhead25.pt" # 保存模型的名字
    # training args:
    # output_path = f'output/SMD/{group}'
    log_dir = f'output/PSM/logs'
    log_tensorboard = True
    writer = SummaryWriter(f"{log_dir}")
    scaler = None
    (x_train, _), (x_test, y_test), _ = get_data("PSM", normalize=True)
    n_features = x_train.shape[1] # 25
    train_dataset = ShiftWindowDataset(x_train, window_size, n_features, output_window)
    test_dataset = ShiftWindowDataset(x_test, window_size, n_features, output_window)
    train_loader, val_loader, = create_data_loaders(  # 获取小批量数据
        train_dataset, batch_size, val_split=0.1, shuffle=False
    )
    # out_dim = n_features
    # 2. build model
    loss_func = nn.MSELoss()
    losses = {
        "train_losses": [],
        "val_losses": [],
    }
    epoch_times = []
    model_transformer = TransEncDec(window_size, output_window, n_features).to("cuda")
    optimizer = torch.optim.Adam(model_transformer.parameters(), lr=init_lr)
    train_start = time.time()
    for e in range(epoch):
        epoch_start = time.time()
        model_transformer.train() #Sets the model in training mode
        train_loss = 0
        train_num = 0
        for src, tgt in tqdm(train_loader):
            # x (b, n, k)
            src = src.to("cuda").to(torch.float32)
            tgt = tgt.to("cuda").to(torch.float32)
            output = model_transformer(src, tgt)
            loss = loss_func(tgt[-model_transformer.output_window:], output[-model_transformer.output_window:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * src.size(0)
            train_num += src.size(0)
        print(f'Epoch{e + 1}/{epoch}: Loss:{train_loss / train_num}')
        e_train_loss = train_loss / train_num
        losses["train_losses"].append(e_train_loss)
        # record the training epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        s = (
            f"[Epoch {epoch + 1}] "
            f"train_loss = {e_train_loss:.5f}, "
        )
        # Evaluate on validation set
        val_loss = "NA"
        if val_loader is not None:
            # 每一次epoch都进行一次验证集上的测试
            val_loss = evaluate_transformer(model_transformer, val_loader)
            if losses["val_losses"] and val_loss <= losses["val_losses"][-1]:
                torch.save(model_transformer.state_dict(), os.path.join(save_model_path, model_name))
            losses["val_losses"].append(val_loss)
        if log_tensorboard: #记录每一epoch的loss
            for key, value in losses.items():
                if len(value) != 0:
                    writer.add_scalar(key, value[-1], epoch)
        if val_loader is not None:
            s += (
                f"val_losses = {val_loss:.5f}"
            )
        s += f" [{epoch_time:.1f}s]"
        print(s)
    if val_loader is None:
        torch.save(model_transformer.state_dict(), os.path.join(save_model_path, model_name))
    train_time = int(time.time() - train_start)
    if log_tensorboard:
        writer.add_text("total_train_time", str(train_time))
    print(f"-- Training done in {train_time}s.")
    """
        model1 = LSTM(input_size=38, hidden_size=128, num_layers=1)
        optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)
        model1 = model1.to("cuda")
        for e in range(epoch):
            train_loss = 0
            train_num = 0
            for x, y in tqdm(train_loader):
                # x (b, n, k)
                x = x.to("cuda")
                y = y.to("cuda")
    
                y_hat = model1(x)
                loss = loss_func(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_num += x.size(0)
            print(f'Epoch{e + 1}/{epoch}: Loss:{train_loss / train_num}')
    """
    # model2 = Encoder_Decoder(window_size+1, n_features, 128, 64) # +1是因为input=cat((x, y), dim=1)
    # optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
    # model2 = model2.to("cuda")
    # for e in range(epoch):
    #     train_loss = 0
    #     train_num = 0
    #     for x, y in tqdm(train_loader):
    #         # x (b, n, k)
    #         x = x.to("cuda")
    #         y = y.to("cuda")
    #         recon_window = model2(x, y) #(b, n+1, k)
    #         loss = loss_func(recon_window[:, -1, :].unsqueeze(1), y) #窗口中最后一个值，和y对应
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item() * x.size(0)
    #         train_num += x.size(0)
    #     print(f'Epoch{e + 1}/{epoch}: Loss:{train_loss / train_num}')

"""
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
        dropout=dropout,
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
        n_epochs=epoch,
        batch_size=batch_size,
        init_lr=init_lr,
        forecast_criterion=forecast_criterion,
        recon_criterion=recon_criterion,
        use_cuda=True,
    )
    trainer.fit(train_loader, val_loader)
"""