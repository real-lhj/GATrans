import os
import pickle

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm
import torch.nn as nn


def normalize_data(data):
    data = np.asarray(data, dtype=np.float32) # np.asarray()默认不copy该对象，除非改变dtype
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data) # 使用0代替数组x中的nan元素
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")
    return data, scaler


def get_splited_data(dataset, test_name, label_name, normalize=True):
    print("load data of:", dataset)
    prefix = "data"
    test_folder = "split_test"
    test_label_folder = "split_test_label"
    df_test = pd.read_csv(os.path.join(prefix, dataset, test_folder, test_name))
    test_data = np.array(df_test.values[0::, 1::])
    df_test_label = pd.read_csv(os.path.join(prefix, dataset, test_label_folder, label_name))
    test_label = np.array(df_test_label.values[0::, 1::])
    df_train = pd.read_csv(os.path.join(prefix, dataset, "train.csv"))
    train_data = np.array(df_train.values[0::, 1::])
    if normalize:
        train_data, scaler = normalize_data(train_data)
        test_data, scaler = normalize_data(test_data)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)
    return (train_data, None), (test_data, test_label), scaler


def get_data(dataset, normalize=True):
    prefix = "data"
    if str(dataset).startswith("machine"):
        prefix += "/ServerMachineDataset/processed"
        print("load data of:", dataset)
        f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
        train_data = pickle.load(f)  # numpy.array
        f.close()
        try:
            f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
            test_data = pickle.load(f)
            f.close()
        except (KeyError, FileNotFoundError):
            test_data = None
        try:
            f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
            test_label = pickle.load(f)
            f.close()
        except (KeyError, FileNotFoundError):
            test_label = None
        if normalize:
            train_data, scaler = normalize_data(train_data)
            test_data, scaler = normalize_data(test_data)
    else:
        # PSM dataset
        print("load data of:", dataset)
        df_train = pd.read_csv(os.path.join(prefix, dataset, "train.csv"))
        train_data = np.array(df_train.values[0::, 1::])
        df_test = pd.read_csv(os.path.join(prefix, dataset, "test.csv"))
        test_data = np.array(df_test.values[0::, 1::])
        df_test_label = pd.read_csv(os.path.join(prefix, dataset, "test_label.csv"))
        test_label = np.array(df_test_label.values[0::, 1::])
        if normalize:
            train_data, scaler = normalize_data(train_data)
            test_data, scaler = normalize_data(test_data)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)

    return (train_data, None), (test_data, test_label), scaler


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim, horizon):
        self.data = data  # get_data()的返回值
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon] # horizon: 模型预测的范围，必须大于1
        return x, y

    def __len__(self):
        return len(self.data) - self.window - self.horizon + 1


class ShiftWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=38, horizon=5):
        self.data = data # get_data()的返回值
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        src = np.append(self.data[index : index + self.window, :][:-self.horizon, :], np.zeros((self.horizon, self.target_dim)), axis=0)
        tgt = self.data[index : index + self.window, :] # horizon: 模型预测的范围
        return src, tgt

    def __len__(self):
        # return -1
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=False):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    else:
        dataset_size = len(train_dataset) #28429=28479-50
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        # if shuffle:
        #     np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # 先根据sampler采样，再组装成batch。因此采样的结果必须形状相同。
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True) #sampler:自定义从数据集中采样的策略
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    # if test_dataset is not None:
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    #     print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader


def plot(eval_model, test_loader, dim, scaler, save_path):
    """
    将真实值和预测值画图
    :param eval_model:
    :param dataset: test_dataset, batch_size=1
    :param scaler:
    :return:
    """
    eval_model.eval()
    preds = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for src, tgt in tqdm(test_loader):
            src = src.to("cuda").to(torch.float32)
            tgt = tgt.to("cuda").to(torch.float32)
            out = eval_model(src, tgt)
            out = out[:, -1, :].cpu() #(b, k)
            preds = torch.cat((preds, out), dim=0)
            truth = torch.cat((truth, tgt[:, -1, :].cpu()), dim=0)
    test_result = scaler.inverse_transform(preds)
    truth = scaler.inverse_transform(truth)
    print(f"test_result.shape:{test_result.shape}, truth.shape:{truth.shape}") # 28479-50
    for k in range(dim):
        print(f"ploting for feature {k}...")
        # 对每一维特征作图并保存
        test_ = test_result[:, k]
        truth_ = truth[:, k]
        fig = plt.figure(1, figsize=(20, 5))
        fig.patch.set_facecolor('xkcd:white')
        plt.plot([m for m in range(len(test_))], test_, color="blue")
        plt.title('Prediction uncertainty')
        plt.plot(truth_, color="black")
        plt.legend(["prediction", "true"], loc="upper left")
        plt.xlabel(f"features {k+1}")
        plt.ylabel("Y")
        # plt.show()
        path = os.path.join(save_path, str(k) + ".png")
        plt.savefig(path)
        plt.clf()  # 更新画布
    return





