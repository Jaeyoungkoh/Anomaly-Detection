import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import gumbel_softmax
from math import sqrt

class channel_mask_generator(torch.nn.Module):
    """
    입력 텐서(B, C, d_model)를 기반으로 동적인 채널 마스크(B, C, C)를 생성합니다.
    """    
    def __init__(self, channels, d_model):
        super(channel_mask_generator, self).__init__()

        self.d_model = d_model
        self.n_vars = channels
        self.generator = nn.Sequential(torch.nn.Linear(self.d_model, self.n_vars, bias=False), nn.Sigmoid())
        with torch.no_grad():
            self.generator[0].weight.zero_()

    def forward(self, x):  # x: (B, C, d_model)

        # 채널 표현 간의 유사도 행렬 계산
        distribution_matrix = self.generator(x)

        # Gumbel-Softmax를 통해 확률을 이진 마스크로 변환
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        # 대각 성분은 항상 1로 설정하여 자기 자신과의 연결은 보장
        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)
        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag

        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape # (B, C, C)

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)

        return resample_matrix
    

class MaskedAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_vars, dropout=0.1, regular_lambda=0.3, temperature=0.1, output_attention=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.scale = 1. / sqrt(self.d_keys)
        self.query_proj = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_proj = nn.Linear(d_model, self.d_keys * n_heads)
        self.value_proj = nn.Linear(d_model, self.d_keys * n_heads)
        self.out_proj = nn.Linear(self.d_keys * n_heads, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention
        self.mask_generator = channel_mask_generator(channels=n_vars, d_model=d_model)
        self.dcloss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)


    def forward(self, queries, keys, values, attn_mask=None):
        x = queries # (B, C, d_model)
        channel_mask = self.mask_generator(x) # (B, C, C)

        B, L, H = x.shape[0], x.shape[1], self.n_heads
        q = self.query_proj(queries).view(B, L, H, -1).transpose(1, 2)
        k = self.key_proj(keys).view(B, L, H, -1).transpose(1, 2)
        v = self.value_proj(values).view(B, L, H, -1).transpose(1, 2)

        scores = torch.einsum("bhie,bhje->bhij", q, k)


        # dcloss 계산을 위한 norm_matrix 생성
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)

        # dcloss 계산
        dcloss_val = self.dcloss(scores, channel_mask, norm_matrix)

        # scores.masked_fill_(channel_mask.unsqueeze(1) == 0, -1e9)
        masked_scores = torch.where(channel_mask.unsqueeze(1) == 0, -1e9, scores)

        attn = self.dropout(torch.softmax(self.scale * masked_scores, dim=-1))
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        
        if self.output_attention:
            return self.out_proj(out), attn, dcloss_val
        else:
            return self.out_proj(out), None, dcloss_val
        

class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k

    def forward(self, scores, attn_mask, norm_matrix):
        # scores shape: (B, H, C, C), attn_mask shape: (B, C, C)
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        # Multi-head의 평균 score를 사용
        eps = 1e-9
        cosine = (scores / (norm_matrix+eps)).mean(1) # (B, C, C)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask
        all_scores = torch.exp(cosine / self.temperature)

        clustering_loss = -torch.log((pos_scores.sum(dim=-1)+eps)/(all_scores.sum(dim=-1)+eps))

        # torch.eye(attn_mask.shape[-1]) : (C,C) 크기의 단위 행렬(Identity Matrix) 생성 > (B,C,C)
        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device) # 가장 Sparce 한 상태(목표)
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(eye.reshape(b, -1) - attn_mask.reshape((b, -1)), p=1, dim=-1)
        loss = clustering_loss.mean(1) + self.k * regular_loss

        mean_loss = loss.mean()

        return mean_loss
    