import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.embed import DataEmbedding_Posi_Temp
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.win_size
        self.pred_len = 0
        self.k = configs.top_k_tn
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model_tn, configs.d_ff_tn,
                            num_kernels=configs.num_kernels_tn),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff_tn, configs.d_model_tn,
                            num_kernels=configs.num_kernels_tn)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                            N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):

    def __init__(self, args):
        super(TimesNet, self).__init__()
        self.seq_len = args.win_size
        self.pred_len = 0
        self.model = nn.ModuleList([TimesBlock(args)
                                    for _ in range(args.e_layers_tn)])
        self.enc_embedding = DataEmbedding_Posi_Temp(args.input_c, args.d_model_tn, args.embed_tn, args.freq, args.dropout_tn)
        self.layer = args.e_layers_tn
        self.layer_norm = nn.LayerNorm(args.d_model_tn)
        self.projection = nn.Linear(args.d_model_tn, args.input_c, bias=True)

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                (stdev[:, 0, :].unsqueeze(1).repeat(
                    1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def forward(self, x_enc):

        dec_out = self.anomaly_detection(x_enc)

        return dec_out, None  # [B, L, D]
