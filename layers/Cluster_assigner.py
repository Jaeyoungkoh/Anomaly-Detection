import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.attn_dual import *
from layers.attention import *


class Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, temp, device, epsilon=0.05):
        '''
        [Step1] 변수 임베딩       : x_emb_ → (bs, n_vars, d_model)
        [Step2] 클러스터 임베딩   : cluster_emb_ → (bs, n_cluster, d_model)
        [Step3] attention 수행     → 결과 (bs, n_cluster, d_model)
        [Step4] batch 평균         → 최종 (n_cluster, d_model)
        '''

        super(Cluster_assigner, self).__init__()
        self.n_vars = n_vars # 변수 개수 
        self.n_cluster = n_cluster # 군집 개수  
        self.d_model = d_model # 모델 차원 (512)
        self.epsilon = epsilon
        self.temp = temp

        self.linear = nn.Linear(seq_len, d_model) # 입력 시계열을 d_model 차원으로 변환 

        # 군집 임베딩 초기화 
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device) #nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))

        # cluster_emb를 외부에서 받지 않고, 내부의 학습 가능한 파라미터로 정의
        # self.cluster_emb = nn.Parameter(torch.empty(self.n_cluster, self.d_model))
        # nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))

        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.p2c = CrossAttention(d_model, n_heads=1)
        
        
    def forward(self, x, cluster_emb):     
        # x: (B, L, C]
        # x_emb: (B, C, d_model)
        # cluster_emb: [K, H] (K: Cluster 수)
        bs = x.shape[0]
        cluster_emb_ = cluster_emb.unsqueeze(0).expand(bs, -1, -1)

        n_vars = x.shape[-1]
        x = x.permute(0,2,1) # (B, C, L)
        x_emb = self.linear(x) # (B, C, L) > (B, C, H) 
        x_emb_flatten = x_emb.reshape(-1, self.d_model)    # (B, C, H) >  (B*C, H]
        # x_emb = x_emb.reshape(-1, self.d_model)
 
        # 군집 할당 확률 계산
        prob = torch.mm(self.l2norm(x_emb_flatten), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster) # (B, C, K)

        # 2. <-- (핵심 변경) Softmax(sinkhorn) 대신 Sigmoid 적용
        # 각 변수-클러스터 쌍에 대해 독립적인 할당 확률(0~1)을 계산합니다.
        # scores = torch.mm(self.l2norm(x_emb), self.l2norm(self.cluster_emb).t())
        # independent_probs = scores.sigmoid()
        # prob_per_batch = independent_probs.reshape(bs, n_vars, self.n_cluster) # (B, C, K)
        # prob_avg = torch.mean(prob_per_batch, dim=0) 

        # prob_temp = prob.reshape(-1, self.n_cluster) #[B*C, K]
        # prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        # prob_temp = prob_temp.reshape(bs, n_vars, self.n_cluster)

        prob_avg = torch.mean(prob, dim=0)    #[C, K]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon) # [C, K] 각 row(feature) 가 softmax처럼 분포가 되도록 정규화
        prob_per_batch = prob_avg.unsqueeze(0).expand(bs, -1, -1)

        # 군집 마스크 생성(Binary Matrix) Reparameterization trick (Gumbel-Sigmoid)
        mask = self.concrete_bern(prob_avg, self.temp)   #[C, K]
        # mask = self.gumbel_softmax_sample(prob_avg, self.temp) #[C, K]

        # logits = torch.log(prob_avg.clamp_min(1e-12))  # [C,K]
        # mask = F.gumbel_softmax(logits=logits, tau=self.temp, hard=False, dim=-1)  # [C,K] one-hot

        mask_per_batch = mask.unsqueeze(0).expand(bs, -1, -1)

        # 군집 중심 벡터 계산
        x_emb_ = x_emb_flatten.reshape(bs, n_vars,-1) #[B, D, H]
        # cluster_emb_ = cluster_emb.repeat(bs,1,1) #[B, K, H]
        cluster_emb = self.p2c(cluster_emb_, x_emb_, x_emb_, mask=mask.transpose(0,1))  # (B, K, H)
        cluster_emb = cluster_emb.mean(dim=0) # [K, H] 모든 batch의 클러스터 임베딩을 평균
    
        return prob_avg, cluster_emb, prob_per_batch, mask_per_batch, x_emb
    
    def concrete_bern(self, prob, temp = 1.0): # concrete 분포를 사용하여 이진 샘플링 수행 
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device) # [D, K] Gumbel trick (reparameterization)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)

        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10) # 확률을 Logit으로 변환 logit(p) = log(p/(1-p))
        prob_bern = ((prob + random_noise) / temp).sigmoid() # sigmoid로 다시 [0,1]범위로 squash. 즉, soft하게 0~1사이 값으로 샘플링된 마스크 생성
        return prob_bern

    def gumbel_softmax_sample(self, logits, temp=1.0):
        # Gumbel(0,1) 노이즈 샘플링
        y = torch.empty_like(logits).exponential_().log()
        y = logits - y

        # Softmax 적용
        return F.softmax(y / temp, dim=-1)

def sinkhorn(out, epsilon=1, sinkhorn_iterations=3):   #[D, K]
    Q = torch.exp(out / epsilon)
    sum_Q = torch.sum(Q, dim=1, keepdim=True) 
    Q = Q / (sum_Q)
    return Q #[D, K]


def cluster_aggregator(var_emb, mask): # 군집 내 변수들의 정보를 집계하는 함수
    '''
        var_emb: (bs*patch_num, nvars, d_model)
        mask: (nvars, n_cluster)
        return: (bs*patch_num, n_cluster, d_model)
    '''
    # x_fused: [B, 3C, L]
    # mask: [3C, K]
    num_var_pc = torch.sum(mask, dim=0)
    var_emb = var_emb.transpose(1,2) # x_fused: [B, L, 3C]
    cluster_emb = torch.matmul(var_emb, mask)/(num_var_pc + 1e-6) # [B,L,K]
    cluster_emb = cluster_emb.transpose(1,2)
    return cluster_emb # [B,K,L]


    
class CrossAttention(nn.Module): # 크로스 어텐션 연산을 수행하는 클래스
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
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        #Q(Cluster_emb):[B, K, H] B 3C L
        #K(Channel emb):[B, D, H] B K L
        #V(Channel emb):[B, D, H] B K L
        #M(Mask): [K, D]
        B, L, _ = queries.shape 
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1) # (B,K,1,H)
        keys = keys.view(B, S, H, -1) # (B,D,1,H)
        values = values.view(B, S, H, -1) # (B,D,1,H) 

        out = self.inner_attention(queries, keys, values, mask) # out = (B, K, 1, H)

        if self.mix:
            out = out.transpose(2,1).contiguous() # (B, 1, K, H)
        out = out.view(B, L, -1) # (B, K, H)

        return out # (B, K, H)
    

class ClusterAttentionLayer(nn.Module):
    """
    fused_output을 d_model 차원으로 투영하고, 
    미리 계산된 cluster_emb를 Key와 Value로 사용하여 어텐션을 수행하는 모듈.
    """
    def __init__(self, args, fused_dim, d_model, n_heads, out_dim, d_ff=None, dropout=0.1, activation="relu"):
        super(ClusterAttentionLayer, self).__init__()
        
        # 1. 입력 프로젝션: (B, C, L) -> (B, C, d_model)
        self.input_projection = nn.Linear(fused_dim, d_model)
        
        # 2. 크로스-어텐션: Q(fused), K(cluster), V(cluster)
        self.cross_attention = AttentionLayer(
            args,  # AttentionLayer의 첫 번째 인자로 args 전달
            attention=FullAttention(
                args,  # FullAttention의 첫 번째 인자로 args 전달
                mask_flag=False,  # 크로스-어텐션이므로 Causal Mask 비활성화
                attention_dropout=dropout,
                output_attention=False
            ),
            d_model=d_model,
            n_heads=n_heads
        )
        
        # 3. Feed Forward 네트워크 (트랜스포머 블록 구조)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff or 4 * d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff or 4 * d_model, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        # 4. 출력 프로젝션:  (B, C, d_model) -> (B, C, L)
        self.output_projection = nn.Linear(d_model, out_dim)

    def forward(self, x_fused, cluster_emb):
        """
        Args:
            x_fused (Tensor): 융합된 피처. shape: [B, C, L]
            cluster_emb (Tensor): 학습된 클러스터 임베딩. shape: [B, K, d_model]
        Returns:
            Tensor: 최종 재구성 결과. shape: [B, L, C]
        """
        # 입력 프로젝션
        x_projected = self.input_projection(x_fused) # (B, C, L) -> (B, C, d_model)
        
        # 크로스-어텐션
        # 정규화 -> 어텐션 -> 드롭아웃 -> 잔차 연결
        # Q: x_projected, K,V: cluster_emb
        norm_x = self.norm1(x_projected)
        attn_out, _ = self.cross_attention(norm_x, cluster_emb, cluster_emb, attn_mask=None) # (B, C, d_model)

        x = x_projected + self.dropout(attn_out)
        # y = x = self.norm1(x)
         
        # Feed Forward
        # 정규화 -> FFN -> 드롭아웃 -> 잔차 연결
        norm_x2 = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(norm_x2.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = x + y # 잔차 연결 2

        # output = self.norm2(x + y) # (B, C, d_model)
        
        # 출력 프로젝션
        final_output = self.output_projection(output) # (B, C, d_model) -> (B, C, L)
        final_output = final_output.permute(0,2,1) # (B, L, C)

        return final_output # (B, L, C)
