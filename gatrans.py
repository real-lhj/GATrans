import torch
import torch.nn as nn
from modules import (
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    ForecastingModel,
    ReconstructionModel,
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

        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = ForecastingModel(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers,
                                               dropout)

    def forward(self, x):
        # x shape (b, n, k)
        h_feat = self.feature_gat(x) #(b, n, k)
        h_temp = self.temporal_gat(x) #(b, n, k)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat) #(b,150)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end) #(b, 38)
        recons = self.recon_model(h_end) #[b, 10, 30]

        return predictions, recons
