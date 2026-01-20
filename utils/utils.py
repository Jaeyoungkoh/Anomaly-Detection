import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def channel_pearson_corr(x, eps=1e-6):
    """
    x: (B, L, C)  — 윈도우 길이 L을 따라 채널 간 상관행렬을 구함
    return: (B, C, C)  — 배치별 채널-채널 피어슨 상관계수
    """
    B, L, C = x.shape
    # (B, L, C) → (B, C, L)
    X = x.transpose(1, 2)
    X = X - X.mean(dim=-1, keepdim=True)                    # zero-mean over time
    cov = X @ X.transpose(-1, -2) / (L - 1 + eps)           # (B, C, C)
    var = torch.diagonal(cov, dim1=-2, dim2=-1)             # (B, C)
    std = torch.sqrt(var.clamp_min(eps))                    # (B, C)
    denom = std.unsqueeze(-1) * std.unsqueeze(-2) + eps     # (B, C, C)
    R = cov / denom
    return R.clamp_(-1.0, 1.0)

def mean_attention_over_heads(attn):
    """
    attn: (B, H, C, C) or (H, C, C)
    return: (B, C, C) with row-normalization (확률 분포)
    """
    if attn.dim() == 3:   # (H,C,C) → (1,H,C,C)
        attn = attn.unsqueeze(0)
    A = attn.mean(dim=1)                  # head 평균: (B, C, C)
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)  # row-stochastic
    return A


def aux_reconstruction_loss(chan_dec_out, x_used, reduction='mean'):
    """
    chan_dec_out: (B, L, C)  — 채널 스트림 복원 결과
    x_used      : (B, L, C)  — 분기에서 실제 사용된 입력 (예: RevIN 적용 후)
    """
    return F.mse_loss(chan_dec_out, x_used, reduction=reduction)


def aux_corr_guidance_kl(attn_ch, x_used, tau=0.2):
    """
    attn_ch : (B, H, C, C)  — 채널 어텐션 가중치(마지막 레이어)
    x_used  : (B, L, C)     — 분기에서 사용된 입력
    KL( A || softmax(R/τ) )  — A, target 모두 row-stochastic
    """
    A = mean_attention_over_heads(attn_ch)                             # (B, C, C)
    R = channel_pearson_corr(x_used)                                   # (B, C, C)

    target = torch.softmax(R / tau, dim=-1)                            # (B, C, C)
    # torch.kl_div의 입력은 log-prob, target은 prob
    loss = F.kl_div((A + 1e-8).log(), target, reduction='batchmean')
    return loss


def aux_corr_guidance_mse(attn_ch, x_used):
    """
    A와 정규화된 R를 직접 MSE로 붙임
    """
    A = mean_attention_over_heads(attn_ch)                 # (B, C, C)
    R = channel_pearson_corr(x_used)                       # (B, C, C)
    Rn = (R - R.mean(dim=-1, keepdim=True)) / (R.std(dim=-1, keepdim=True) + 1e-6)
    # Rn을 row-softmax로 바꿔서 분포형으로 맞추는 것도 가능:
    # Rn = torch.softmax(R / 0.2, dim=-1)
    return F.mse_loss(A, Rn, reduction='mean')

def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    
    trainer.losses = {
        "train_total": [],
        "train_forecast": [],
        "train_recon": [],
        "val_total": [],
        "val_forecast": [],
        "val_recon": [],
    }
    """

    plt.plot(losses["train_loss"], label="Train loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_loss"], label="Val loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()