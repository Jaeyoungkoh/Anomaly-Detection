import torch
import torch.nn as nn

from layers.layers import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel
)

from layers.normalization import RevIN, DishTS
from layers.Cluster_assigner import *

class MTAD_GAT(nn.Module):
    def __init__(self, args):
          super(MTAD_GAT, self).__init__()

          self.n_features = args.input_c 
          self.window_size = args.win_size # Lookback
          self.out_dim = args.input_c 
          self.kernel_size = args.kernel_size
          self.feat_gat_embed_dim = args.feat_gat_embed_dim # [FEATURE-oriented GAT layer]의 임베딩 차원
          self.time_gat_embed_dim = args.time_gat_embed_dim # [TIME-oriented GAT layer]의 임베딩 차원
          self.use_gatv2 = args.use_gatv2 
          self.gru_n_layers = args.gru_n_layers
          self.gru_hid_dim = args.gru_hid_dim
          self.model_type = args.model_type
          self.forecast_n_layers=args.forecast_n_layers
          self.forecast_hid_dim=args.forecast_hid_dim
          self.recon_n_layers=args.recon_n_layers
          self.recon_hid_dim=args.recon_hid_dim

          self.dropout=args.dropout_gat
          self.alpha=args.alpha

          self.device = "cuda" if torch.cuda.is_available() else "cpu"

          self.norm_type = args.norm_type
          self.affine = args.affine
          self.subtract_last = args.subtract_last

          self.conv = ConvLayer(self.n_features, self.kernel_size)
          self.feature_gat = FeatureAttentionLayer(self.n_features, self.window_size, self.dropout, self.alpha, self.feat_gat_embed_dim, self.use_gatv2)
          self.temporal_gat = TemporalAttentionLayer(self.n_features, self.window_size, self.dropout, self.alpha, self.time_gat_embed_dim, self.use_gatv2)
          self.gru = GRULayer(3 * self.n_features, self.gru_hid_dim, self.gru_n_layers, self.dropout)
          self.forecasting_model = Forecasting_Model(self.gru_hid_dim, self.forecast_hid_dim, self.out_dim, self.forecast_n_layers, self.dropout)
          # self.forecasting_model = Forecasting_Model(self.n_features * 3, self.forecast_hid_dim, self.out_dim, self.forecast_n_layers, self.dropout)

          self.recon_model = ReconstructionModel(self.window_size, self.gru_hid_dim, self.recon_hid_dim, self.out_dim, self.recon_n_layers, self.dropout)

          if self.norm_type == 'revin':
            self.norm = RevIN(num_features=self.n_features, affine=self.affine, subtract_last=self.subtract_last)
          if self.norm_type == 'dish-ts':
            self.norm = DishTS(n_series=self.n_features, seq_len=self.window_size, dish_init='uniform')
            print(f'Normalization for distribution shift using { self.norm_type}')

    def forward(self, x):
        # X의 차원 : (B,L,D)

      # (1) 1Dconv 임베딩 (kernel=7)         
      x_conv = self.conv(x) # (B,L,D)
  
      # (2) 2개의 GAT layer들 거치기 ( + concatenate )
      h_feat, attn = self.feature_gat(x_conv) # (B,L,D)
      h_temp = self.temporal_gat(x_conv) # (B,L,D)
      
      # (3) Concatenation
      h_cat = torch.cat([x_conv, h_temp, h_feat], dim=2) # (B,L,3D)

      # (4) GRU
      _, h_end = self.gru(h_cat) # h_end : (B, H)
      h_end = h_end.view(x.shape[0], -1) # Hidden state 펼치기 (B, H)

      if self.model_type == 'reconstruction':
        recons = self.recon_model(h_end) # (B, L, D)

        return recons, None, attn


      elif self.model_type == 'mix':
        recons = self.recon_model(h_end) # (B, L, D)
        predictions = self.forecasting_model(h_end) # (B, D)
      
        return recons, predictions, attn

      