import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, args, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = True
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).to(args.device)
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, args, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()
        
        self.args = args
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma


class VariableTemporalAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.variable_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.variable_to_out = nn.Linear(inner_dim, dim)
        self.variable_dropout = nn.Dropout(dropout)

        self.temporal_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.temporal_to_out = nn.Linear(inner_dim, dim)
        self.temporal_dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, x, use_attn):
        b, t, f, d = x.shape

        vx = rearrange(x, 'b t f d -> (b t) f d')
        h = self.heads
        vq, vk, vv = self.variable_to_qkv(vx).chunk(3, dim=-1)
        vq, vk, vv = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (vq, vk, vv))
        sim = einsum('b h i d, b h j d -> b h i j', vq, vk) * self.scale

        vattn = sim.softmax(dim=-1)
        if use_attn:
            vweights = vattn
        else:
            vweights = None
        vattn = self.variable_dropout(vattn)

        vout = einsum('b h i j, b h j d -> b h i d', vattn, vv)
        vout = rearrange(vout, 'b h n d -> b n (h d)', h=h)
        vout = self.variable_to_out(vout)
        vout = rearrange(vout, '(b t) f d -> b t f d', t=t)

        tx = rearrange(x, 'b t f d -> (b f) t d')
        tq, tk, tv = self.temporal_to_qkv(tx).chunk(3, dim=-1)
        tq, tk, tv = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (tq, tk, tv))
        sim = einsum('b h i d, b h j d -> b h i j', tq, tk) * self.scale

        tattn = sim.softmax(dim=-1)
        if use_attn:
            tweights = tattn
        else:
            tweights = None
        tattn = self.temporal_dropout(tattn)

        tout = einsum('b h i j, b h j d -> b h i d', tattn, tv)
        tout = rearrange(tout, 'b h n d -> b n (h d)', h=h)
        tout = self.temporal_to_out(tout)
        tout = rearrange(tout, '(b f) t d -> b t f d', f=f)

        out = torch.cat([vout, tout], dim=-1)
        out = self.linear(out)
        return out, vweights, tweights
    

class VariableAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn):
        b, t, f, d = x.shape
        x = rearrange(x, 'b t f d -> (b t) f d')
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = rearrange(out, '(b t) f d -> b t f d', t=t)
        return out, weights


class TemporalAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn):
        b, t, f, d = x.shape
        x = rearrange(x, 'b t f d -> (b f) t d')
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = rearrange(out, '(b f) t d -> b t f d', f=f)
        return out, weights
