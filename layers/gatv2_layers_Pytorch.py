import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gatv2_conv_Pytorch import GATv2Pytorch

class Pytorch_FeatureAttention(nn.Module):

    def __init__(self, 
                 in_channels,               # Window Size (L)
                 out_channels,              # GAT Hidden Dimension (D)
                 num_layers,
                 heads=1,                   # GAT Layer 반복 횟수 (graph_model.py 스타일)
                 concat=True, 
                 dropout=0.1, 
                 negative_slope=0.2, 
                 bias=True, 
                 use_residual=True,   
                 use_layer_norm=True, 
                 use_activation=True          
                 ):
        super(Pytorch_FeatureAttention, self).__init__()        

        self.in_channels = in_channels
        self.heads = heads
        self.concat = concat
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.use_activation = use_activation
        self.dropout_p = dropout 

        self.hidden_dim = out_channels if out_channels is not None else in_channels

        # Hidden Dimension 계산
        if concat:
            # 전체 차원이 head 개수로 나누어 떨어지는지 확인
            assert self.hidden_dim % heads == 0, \
                f"Hidden dimension({self.hidden_dim}) must be divisible by heads({heads})"
            self.head_dim = self.hidden_dim // heads
        else:
            # Concat하지 않는 경우(Average) Head 차원이 곧 전체 차원
            self.head_dim = self.hidden_dim

        # 1. Embedding Layer
        self.embedding = nn.Linear(in_channels, self.hidden_dim)

        # 2. GNN Layers (ModuleList)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # graph_model.py 스타일: Layer 생성
            self.layers.append(
                GATv2Pytorch(
                    in_channels=self.hidden_dim,    # 전체 차원 입력
                    out_channels=self.head_dim,     # Head 당 차원
                    heads=heads, 
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=dropout, 
                    bias=bias
                )
            )
            # 결과가 Concat 되면: head_dim * heads = hidden_dim (원복됨)
            if self.use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.hidden_dim))

        # 3. Output Projection (Hidden -> L)
        # 원래 시계열 길이로 복원
        self.proj = nn.Linear(self.hidden_dim, in_channels)

    def forward(self, x):
        """
        x: (Batch, Length, Channel)
        """
        B, L, C = x.shape
        
        # 1. PyG 입력을 위해 차원 변경 (Node = Channel, Feature = Time Length)
        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1) 
        x = self.embedding(x)
        
        # 2. Layer Loop
        attns = None # 마지막 레이어의 Attention Map 저장용
        for i in range(self.num_layers):
            # (1) GAT Conv
            # 마지막 레이어에서만 Attention Weight를 가져옴 (Visualization용)
            return_attn = (i == self.num_layers - 1)
            
            if return_attn:
                new_x, attns = self.layers[i](x, return_attention_weights=True)
            else:
                new_x = self.layers[i](x, return_attention_weights=False)

            # (2) Activation
            if self.use_activation:
                new_x = F.elu(new_x)
            
            new_x = F.dropout(new_x, p=self.dropout_p, training=self.training)

            # (3) Residual Connection
            # 차원이 맞을 때만 수행 (첫 레이어 등에서 차원 다르면 skip하거나 projection 필요하지만, 
            # 여기선 Embedding으로 미리 맞춰둠)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            
            # (4) Layer Norm
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
        # 3. Output Projection (Hidden -> L)
        # (B, C, Hidden) -> (B, C, L)
        out_flat = self.proj(x)
        
        # Restore Shape: (B, C, L) -> (B, L, C)
        out = out_flat.view(B, C, L).permute(0, 2, 1)

        return out, attns