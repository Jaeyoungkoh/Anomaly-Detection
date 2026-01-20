import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attn_dual import FullAttention, AttentionLayer
from layers.embed import DataEmbedding, TokenEmbedding, PositionalEmbedding
from layers.normalization import RevIN
from layers.layers import ConvLayer,CausalConvLayer, Forecasting_Model, FeatureAttentionLayer, MultiHeadFeatureAttention

class ChannelEmbedding(nn.Module):
    """
    Channel-wise 입력을 위한 임베딩 클래스.
    입력 shape: (B, C, L)
    (B, C, L) -> (B, C, d_model)로 변환합니다.
    """
    def __init__(self, c_in, d_model, dropout=0.0):
        super(ChannelEmbedding, self).__init__()
        # c_in은 이제 채널 수(C)가 아니라 시퀀스 길이(L)가 됩니다.
        # 각 채널의 L 길이 시계열을 d_model 차원의 벡터로 변환
        self.value_embedding = nn.Linear(c_in, d_model, bias=False)
        # Positional Embedding은 채널의 위치를 인코딩하는데 재사용
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (B, C, L)
        # value_embedding을 통과시키면 (B, C, d_model)
        # position_embedding은 (B, C, d_model)의 C 차원에 대한 위치 정보를 더해줌
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):

        # Multi-Head Attention (Pre-LN)
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        # Residual            
        x = x + self.dropout(new_x)
        # LN1
        y = x = self.norm1(x) # B, L, D
        # FF            
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # Residual            
        x = x + y
        # LN2            
        x = self.norm2(x)

        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
 
        for attn_layer in self.attn_layers:
            x, series = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list


class Proposed(nn.Module):
    def __init__(self, args):

        super(Proposed, self).__init__()

        self.output_attention = args.output_attention
        self.enc_in = args.input_c
        self.enc_out = args.input_c
        self.d_model = args.d_model_temp
        self.dropout_temp = args.dropout_temp
        self.dropout_gat = args.dropout_gat
        self.dropout_fore = args.dropout_fore
        self.alpha = args.alpha
    
        self.feat_gat_embed_dim = args.d_model_gat
        self.gat_type = args.gat_type
        self.use_node_embedding = args.use_node_embedding
        self.win_size = args.win_size
        self.n_heads = args.n_heads_temp
        self.n_heads_gat = args.n_heads_gat
        self.e_layers = args.e_layers_temp
        self.d_ff_temp = args.d_ff_temp
        self.d_ff_channel = args.d_ff_channel
        self.activation_temp = 'gelu'
        self.activation_chan = 'gelu'
        self.concat = args.concat

        self.forecast_hid_dim = args.fore_hid_dim
        self.forecast_n_layers = args.fore_n_layers

        self.kernel_size = args.kernel_size

        self.norm_type = args.norm_type
        self.affine = args.affine
        self.subtract_last = args.subtract_last
        self.use_gatv2 = args.use_gatv2

        if self.norm_type == 'revin':
            self.norm = RevIN(num_features=self.enc_in, affine=self.affine, subtract_last=self.subtract_last)

        # self.conv = ConvLayer(self.enc_in, self.kernel_size)
        self.causal_conv = CausalConvLayer(self.enc_in, self.kernel_size)

        # --- Branch 1: Temporal Attention Stream ---
        self.temporal_embedding = DataEmbedding(self.enc_in, self.d_model, self.dropout_temp)
        self.temporal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=self.dropout_temp, output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model, self.d_ff_temp, dropout=self.dropout_temp, activation=self.activation_temp
                ) for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.temporal_decoder = nn.Linear(self.d_model, self.enc_out)

        # --- Branch 2: Channel Attention Stream ---

        # self.feature_gat = FeatureAttentionLayer(self.enc_in, self.win_size, self.dropout_gat, self.alpha, self.feat_gat_embed_dim, self.use_gatv2)
        self.feature_gat = FeatureAttentionLayer(self.enc_in, self.win_size, self.dropout_gat, self.alpha, self.feat_gat_embed_dim, self.use_gatv2)

        # --- Fusion Layer ---
        self.forecasting_model = Forecasting_Model(self.enc_out * 3, self.forecast_hid_dim, self.enc_out, self.forecast_n_layers, self.dropout_fore)

    def forward(self, x):

        attns = {} # 시각화 데이터를 담을 딕셔너리
        # x shape: [B, L, C]

        # x_conv = self.conv(x) # (B,L,C)
        x_conv = self.causal_conv(x) # (B,L,C)

        if self.norm_type == 'revin':
            x = self.norm(x, 'n')

        # --- Branch 1: Temporal Attention ---
        temp_embed_out = self.temporal_embedding(x)
        temp_enc_out, temp_series_list = self.temporal_encoder(temp_embed_out) # (B, L, d_model)
        temp_dec_out = self.temporal_decoder(temp_enc_out) # (B, L, d_model) -> (B, L, C)
    
        # --- Branch 2: Channel Attention ---
        chan_enc_out, chan_series_list = self.feature_gat(x) # (B, d_model, C)
        # chan_enc_out = chan_enc_out.permute(0, 2, 1) # (B, C, d_model)
        # chan_dec_out = self.channel_decoder(chan_enc_out).permute(0, 2, 1) # (B, C, L)

        # chan_enc_out, chan_series_list = self.channel_encoder(x) # # (B, C, feat_gat_embed_dim = L)
        # chan_dec_out = self.channel_decoder(chan_enc_out) # (B, C, L)
        # chan_dec_out = chan_dec_out.permute(0, 2, 1) # (B, L, C)

        # --- Fusion ---
        fused_output = torch.cat([temp_dec_out, chan_enc_out, x_conv], dim=-1) # (B, L, 3C)
        final_output = self.forecasting_model(fused_output)
        
        if self.output_attention:
            # 두 어텐션을 딕셔너리 형태로 반환하거나 필요에 맞게 가공
            attns['temporal'] = temp_series_list[-1]
            attns['channel'] = chan_series_list

            return final_output, attns
        else:
            return final_output, None
