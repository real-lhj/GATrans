import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            window_size,
            n_features,
            target_dims=None,
            n_epochs=200,
            batch_size=256,
            init_lr=0.0001,
            forecast_criterion=nn.MSELoss(),
            recon_criterion=nn.MSELoss(),
            use_cuda=True,

    ):
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
        }
        self.epoch_times = []
        if self.device == "cuda":
            self.model.cuda()


    def fit(self, train_loader, val_loader=None):
        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []

            for x, y in tqdm(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                preds, recons = self.model(x, y)  # 预测结果

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)
                #recons.shape = (16,10,114)
                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds)) # criterion损失函数
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                loss = forecast_loss + recon_loss
                loss.backward()  # 反向传播
                self.optimizer.step()  # 优化参数 Adam优化器

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)
            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())
            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time) # 每个epoch的所需时间
            if epoch % 1 == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )
                s += f" [{epoch_time:.1f}s]"
                print(s)
        train_time = int(time.time() - train_start)
        print(f"-- Training done in {train_time}s.")




