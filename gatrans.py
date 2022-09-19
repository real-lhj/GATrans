import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from modules import (
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    ForecastingModel,
    ReconstructionModel,
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
class GATrans(nn.Module):
    def __init__(
            self,
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
    ):
        super(GATrans, self).__init__()
        self.feats = n_features
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = ForecastingModel(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)
        self.pos_encoder = PositionalEncoding(n_features, window_size)
        encoder_layers = TransformerEncoderLayer(d_model=n_features, nhead=n_features, dim_feedforward=16, dropout=0.2)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=n_features, nhead=n_features, dim_feedforward=16, dropout=0.2)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sigmoid() # 输出做归一化处理

    def forward(self, x, y):
        # x shape (b, n, k)

        h_feat = self.feature_gat(x) #(b, n, k)
        h_temp = self.temporal_gat(x) #(b, n, k)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat) #(b,150)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end) #(b, 38)
        recons = self.recon_model(h_end) #[b, 10, 30]

        return predictions, recons

        """
        h_feat = self.feature_gat(x) # (b, n, k)
        src = h_feat.permute(1, 0, 2) #(n, b, k)
        tgt = y.permute(1, 0, 2)
        src = src * math.sqrt(self.feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        y = self.transformer_decoder(tgt, memory).permute(1, 0, 2)
        y = self.fcn(y)
        return y
        """








