import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attn import AnomalyAttention, AttentionLayer
from layers.embed import DataEmbedding, TokenEmbedding


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
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # FFN(Feed-Forward Networks)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, args):

        super(AnomalyTransformer, self).__init__()

        self.output_attention = args.output_attention
        self.enc_in = args.input_c
        self.enc_out = args.input_c
        self.d_model = args.d_model # 512
        self.dropout = args.dropout
        self.win_size = args.win_size
        self.n_heads = args.n_heads # 8
        self.e_layers = args.e_layers # 3
        self.d_ff = args.d_ff
        self.activation = 'gelu'

        # Encoding
        self.embedding = DataEmbedding(self.enc_in, self.d_model, self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(args,
                        AnomalyAttention(args, self.win_size, False, attention_dropout=self.dropout, output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projection = nn.Linear(self.d_model, self.enc_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
