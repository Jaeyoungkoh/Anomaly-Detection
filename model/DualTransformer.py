import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attn_dual import FullAttention, AttentionLayer
from layers.embed import DataEmbedding, TokenEmbedding, PositionalEmbedding
from layers.normalization import RevIN
from layers.layers import ConvLayer,CausalConvLayer, Forecasting_Model, ReconstructionModel, GRULayer
from utils.utils import channel_pearson_corr, mean_attention_over_heads, aux_reconstruction_loss, aux_corr_guidance_kl


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
        # # 1. Multi-Head Attention (Pre-LN)
        # norm_x = self.norm1(x)
        # # new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        # # x = x + self.dropout(new_x)
        # new_x, attn = self.attention(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        # x = x + self.dropout(new_x)

        # # 2. Feed Forward Network (Pre-LN)
        # norm_x2 = self.norm2(x)
        # y = self.dropout(self.activation(self.conv1(norm_x2.transpose(-1, 1))))

        # # y = x = self.norm1(x)
        # # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # # y = self.dropout(self.conv2(y).transpose(-1, 1))

        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # x = x + y

        # # return self.norm2(x + y), attn
        # return x, attn

        # 1. Multi-Head Attention (Pre-LN)
        # norm_x = self.norm1(x)
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        # new_x, attn = self.attention(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        # 2. Feed Forward Network (Pre-LN)
        # norm_x2 = self.norm2(x)
        # y = self.dropout(self.activation(self.conv1(norm_x2.transpose(-1, 1))))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y

        return self.norm2(x + y), attn
        # return x, attn



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


class DualTransformer(nn.Module):
    def __init__(self, args):

        super(DualTransformer, self).__init__()

        self.output_attention = args.output_attention
        self.enc_in = args.input_c
        self.enc_out = args.input_c
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.win_size = args.win_size
        self.n_heads = args.n_heads
        self.e_layers = args.e_layers
        self.d_ff = args.d_ff
        self.activation = 'gelu'

        self.forecast_hid_dim = args.fore_hid_dim
        self.forecast_n_layers = args.fore_n_layers

        self.kernel_size = args.kernel_size

        self.norm_type = args.norm_type
        self.affine = args.affine
        self.subtract_last = args.subtract_last
        
        if self.norm_type == 'revin':
            self.norm = RevIN(num_features=self.enc_in, affine=self.affine, subtract_last=self.subtract_last)

        # self.conv = ConvLayer(self.enc_in, self.kernel_size)
        self.causal_conv = CausalConvLayer(self.enc_in, self.kernel_size)
        
        # --- Branch 1: Temporal Attention Stream ---
        self.temporal_embedding = DataEmbedding(self.enc_in, self.d_model, self.dropout)
        self.temporal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(args,
                        FullAttention(args, mask_flag=False, attention_dropout=self.dropout, output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model, self.d_ff, dropout=self.dropout, activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # --- Branch 2: Channel Attention Stream ---
        self.channel_embedding = ChannelEmbedding(self.win_size, self.d_model, self.dropout) 
        self.channel_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(args,
                        FullAttention(args, mask_flag=False, attention_dropout=self.dropout, output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model, self.d_ff, dropout=self.dropout, activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        ) 
        self.channel_decoder = nn.Linear(self.d_model, self.win_size)
        self.temporal_decoder = nn.Linear(self.d_model, self.enc_out)

        # --- Fusion Layer ---
        self.forecasting_model = Forecasting_Model(self.enc_out * 3, self.forecast_hid_dim, self.enc_out, self.forecast_n_layers, self.dropout)

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
        x_transposed = x.permute(0, 2, 1) # (B, C, L)
        chan_embed_out = self.channel_embedding(x_transposed) # (B, C, d_model)
        chan_enc_out, chan_series_list = self.channel_encoder(chan_embed_out) # (B, C, d_model)
        chan_dec_out = self.channel_decoder(chan_enc_out) # (B, C, d_model) -> (B, C, L)
        chan_dec_out = chan_dec_out.permute(0, 2, 1) # (B, L, C)

        # --- Fusion ---
        fused_output = torch.cat([chan_dec_out, temp_dec_out, x_conv], dim=-1) # (B, L, 3C)
        final_output =  self.forecasting_model(fused_output)
        
        if self.output_attention:
            # 두 어텐션을 딕셔너리 형태로 반환하거나 필요에 맞게 가공
            attns['temporal'] = temp_series_list[-1]
            attns['channel'] = chan_series_list[-1]
            attns['chan_dec_out'] = chan_dec_out              # ★ 보조 재구성 손실용 (B, L, C)
            attns['x_used_in_branches'] = x                   # ★ RevIN 적용 후 x (B, L, C)
            attns['channel_emb_per_batch'] = chan_embed_out

            return final_output, attns
        else:
            return final_output, None
