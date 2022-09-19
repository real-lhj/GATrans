import json
import torch
from tqdm import tqdm
from utils import *
from eval_methods import *
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class Predictor():
    def __init__(self, model, window_size, output_window, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.output_window = output_window
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 128
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name
        self.scaler = pred_args["scaler"]

    def scale_scores(self, a_score):
        q75, q25 = np.percentile(a_score, [75, 25])
        iqr = q75 - q25
        median = np.median(a_score)
        a_score = (a_score - median) / (1 + iqr)
        return a_score

    def get_score(self, values, scaler):
        """
        :param values: [n, 38], n is the num of samples, 38 is the feature num(or embed_dim)
        :return:
        """
        print("Predicting and calculating anomaly scores..")
        data = ShiftWindowDataset(values, self.window_size, self.n_features, horizon=self.output_window)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model = self.model.to(device)
        preds = torch.Tensor(0)
        recons = []
        if self.model.model_name == "transformer":
            # actual = values[self.window_size + self.output_window - 1:, :]
            # anomaly_scores = np.zeros_like(actual)
            # flag = torch.Tensor(0)
            with torch.no_grad():
                for src, tgt in tqdm(loader):
                    src = src.to("cuda").to(torch.float32)
                    tgt = tgt.to("cuda").to(torch.float32)
                    out = self.model(src, tgt)
                    out = out[:, -1, :].cpu()  # (b, k)
                    preds = torch.cat((preds, out), dim=0)
                    # flag = torch.cat((flag, tgt[:, -1, :].cpu()), dim=0)
            preds = scaler.inverse_transform(preds)
            actual = values[self.window_size - 1:, :][:preds.shape[0], :]
            actual = scaler.inverse_transform(actual)
            anomaly_scores = np.zeros_like(actual)
            df = pd.DataFrame()
            for i in range(preds.shape[1]):
                a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2)  # 差
                anomaly_scores[:, i] = a_score
                df[f"A_Score_{i}"] = a_score
            entity_anomaly_scores = np.mean(anomaly_scores, 1)
            df['A_Score_Global'] = entity_anomaly_scores
            return df
        """
        elif self.model.model_name == "lstm":
            with torch.no_grad():
                for x, y in tqdm(loader):
                    x = x.to(device)
                    y_hat = self.model(x)
                    preds.append(y_hat.squeeze(1).detach().cpu().numpy())  # [256,1,38]=>[256,38]
            preds = np.concatenate(preds, axis=0)
            df = pd.DataFrame()
            for i in range(preds.shape[1]):
                a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2)  # 差
                if self.scale_scores:
                    a_score = self.scale_scores(a_score)
                anomaly_scores[:, i] = a_score
                df[f"A_Score_{i}"] = a_score
            entity_anomaly_scores = np.mean(anomaly_scores, 1)
            df['A_Score_Global'] = entity_anomaly_scores
            return df

        elif self.model.model_name == "encoder_decoder":
            with torch.no_grad():
                for x, y in tqdm(loader):
                    x = x.to(device)
                    y = y.to(device)
                    recon_window = self.model(x, y)
                    recon_y = recon_window[:, -1, :]
                    recons.append(recon_y.squeeze(1).detach().cpu().numpy())
            recons = np.concatenate(recons, axis=0)
            df = pd.DataFrame()
            for i in range(recons.shape[1]):
                a_score = np.sqrt((recons[:, i] - actual[:, i]) ** 2)  # 差
                if self.scale_scores:
                    a_score = self.scale_scores(a_score)
                anomaly_scores[:, i] = a_score
                df[f"A_Score_{i}"] = a_score
            entity_anomaly_scores = np.mean(anomaly_scores, 1)
            df['A_Score_Global'] = entity_anomaly_scores
            return df
        """

    def predict_anomalies(self, train, test, true_anomalies, load_scores=False, save_output=True,
                          scale_scores=False):
        train_pred_df = self.get_score(train, self.scaler)  # train的score #A_Score_Global:(len(train),)是每一行的平均值
        test_pred_df = self.get_score(test, self.scaler)  # test的score
        true_anomalies = true_anomalies[:len(test_pred_df)]
        # entity的异常分数
        train_anomaly_scores = train_pred_df['A_Score_Global'].values
        test_anomaly_scores = test_pred_df['A_Score_Global'].values

        # Find threshold and predict anomalies at feature-level (for plotting and diagnosis purposes)
        out_dim = self.n_features
        all_preds = np.zeros((len(test_pred_df), out_dim))
        # for i in range(out_dim):
        #     train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
        #     test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
        #     epsilon = find_epsilon(train_feature_anom_scores, reg_level=2)
        #     train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(int)
        #     test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)
        #     train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
        #     test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds
        #
        #     all_preds[:, i] = test_feature_anom_preds
        # Global anomalies (entity-level) are predicted using aggregation of anomaly scores across all features（所有列特征的score的平均值）
        e_eval = epsilon_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies, reg_level=self.reg_level)
        p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,
                          q=self.q, level=self.level, dynamic=self.dynamic_pot)
        # if true_anomalies is not None:
        #     # 只用测试集，暴力搜索寻找阈值
        #     bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0.01, end=2, step_num=100, verbose=False)
        # else:
        #     bf_eval = {}

        print(f"Results using epsilon method:\n {e_eval}")
        print(f"Results using peak-over-threshold method:\n {p_eval}")
        # print(f"Results using best f1 score search:\n {bf_eval}")



