import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

class MaskAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(MaskAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout) # 드롭아웃을 적용하여 과적합 방지 
        
    def forward(self, queries, keys, values, mask=None):
        # HEAD=1
        # Q(B,K,1,H)
        # K(B,D,1,H)    
        # V(B,D,1,H)
        # M(K,D)

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E) # 스케일리 값이 지정되지 않으면 E의 제곱근의 역수를 사용 

        # 어텐션 점수 계산 (batch-wise dot product)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # (B, 1, K, D) 
        
        # # 소프트맥스를 적용한 후 드롭아웃 수행 
        A = self.dropout(torch.softmax(scale * scores, dim=-1)) # (B, 1, K, D) 

        # # 마스크가 있는 경우 마스크를 적용하여 특정 위치의 어텐션을 제한 
        A = A if mask == None else A * mask # (B, 1, K, D) * (K, D) Broadcasting -> (B, 1, K, D)

        # --- 올바른 마스킹 적용 위치 ---
        # 마스크가 있는 경우, softmax 이전에 마스크를 적용
        # if mask is not None:
             # mask에서 0인 위치 (마스킹할 위치)에 매우 작은 값을 채워넣음
             # 이렇게 하면 softmax 후 해당 위치의 값은 0에 수렴하게 됨
        #     scores = scores.masked_fill(mask == 0, -1e9) 
        
        # 스케일링 후 소프트맥스를 적용하여 어텐션 가중치 계산
        # attn_weights = torch.softmax(scale * scores, dim=-1) # (B, H, L, S)
        
        # A = self.dropout(attn_weights)

        # 최종 어텐션 결과를 값 벡터와 곱하여 최종 출력 계산
        V = torch.einsum("bhls,bshd->blhd", A, values) # (B, K, 1, H)
        
        return V.contiguous() # (B, K, 1, H) 연속적인 텐서 반환

class MaskAttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)

    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(MaskAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        # 내부 마스크 어텐션 레이어 정의 
        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        
        # 입력 차원에서 멀티헤드 차원으로 변환하는 프로젝션 레이어 정의 
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model) # 출력 차원 변환 
        self.n_heads = n_heads # 헤드 개수 
        self.mix = mix # 헤드 간 혼합 여부 설정 

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 쿼리, 키, 값에 대해 선형 변환 후 멀티헤드 차원으로 변환 
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 내부 어텐션 연산 수행
        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )

        # mix 옵션이 활성화된 경우 차원 변환 수행 
        if self.mix:
            out = out.transpose(2,1).contiguous()

        # 최종 출력을 원래 차원으로 복원 
        out = out.view(B, L, -1)

        return self.out_projection(out) # B, L, d_model, 최종 출력 반환 
