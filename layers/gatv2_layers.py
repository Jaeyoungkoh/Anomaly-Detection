import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gatv2_conv_PyG import GATv2Conv
from torch_geometric.utils import to_dense_adj

class PyG_FeatureAttention(nn.Module):
    def __init__(self, 
                 in_channels,               # Window Size (L)
                 out_channels,              # GAT Hidden Dimension (D)
                 num_layers,
                 heads=1,                   # GAT Layer 반복 횟수 (graph_model.py 스타일)
                 concat=True, 
                 dropout=0.1, 
                 negative_slope=0.2, 
                 add_self_loops=True, 
                 bias=True, 
                 share_weights=False,
                 use_residual=True,   
                 use_layer_norm=True, 
                 use_activation=True          
                 ):
        super().__init__()
        
        # in_channels: window_size (L) - 노드(채널)가 가진 feature의 크기
        # out_channels: GAT hidden dimension

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

        # 1. Embedding Layer (graph_model.py의 Embedding 대체)
        # 시계열 데이터(Continuous)이므로 nn.Embedding 대신 nn.Linear 사용
        # (L -> Hidden Dim)
        self.embedding = nn.Linear(in_channels, self.hidden_dim)
        # self.emb_activation = nn.ReLU() # graph_model.py의 layer0_ff

        # 2. GNN Layers (ModuleList)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # graph_model.py 스타일: Layer 생성
            self.layers.append(
                GATv2Conv(
                    in_channels=self.hidden_dim,    # 전체 차원 입력
                    out_channels=self.head_dim,     # Head 당 차원
                    heads=heads, 
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=dropout, 
                    add_self_loops=add_self_loops,
                    bias=bias,
                    share_weights=share_weights
                )
            )
            # 결과가 Concat 되면: head_dim * heads = hidden_dim (원복됨)
            if self.use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.hidden_dim))

        # 3. Output Projection (Hidden -> L)
        # 원래 시계열 길이로 복원
        self.proj = nn.Linear(self.hidden_dim, in_channels)


    def forward(self, x, edge_index):
        """
        x: (Batch, Length, Channel) -> Proposed_v2에서 넘어오는 형태
        edge_index: (Batch, 2, E) 또는 (2, E) -> Dataloader가 주는 형태
        """
        B, L, C = x.shape
        device = x.device
        
        # 1. PyG 입력을 위해 차원 변경 (Node = Channel, Feature = Time Length)
        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1) 
        
        # Flatten: (Batch * Channel, Length) -> PyG는 (N, D) 형태를 받음
        x_flat = x.reshape(B * C, L)
        # x = x.reshape(B * C, L)
        
        # 2. Embedding Step (L -> Hidden)
        x = self.embedding(x_flat)
        # x = self.emb_activation(x_emb) # Initial Feature H0

        # 2. edge_index 배칭 (Batch Size만큼 확장하여 하나의 큰 Disjoint Graph 생성)
        # edge_index는 (2, E) 형태여야 함.
        # dataloader가 (B, 2, E)로 준다면 첫번째 것을 사용 (Graph 구조가 고정적이라 가정)
        if edge_index.dim() == 3:
            target_edge_index = edge_index[0]
        else:
            target_edge_index = edge_index

        # (2, E) -> (2, B*E) 배치 확장
        batch_edge_index = get_batch_edge_index(target_edge_index, B, C).to(device)

        # 3. Layer Loop
        attn_map = None # 마지막 레이어의 Attention Map 저장용
        for i in range(self.num_layers):
            # (1) GAT Conv
            # 마지막 레이어에서만 Attention Weight를 가져옴 (Visualization용)
            return_attn = (i == self.num_layers - 1)
            
            if return_attn:
                new_x, (edge_idx_ret, alpha) = self.layers[i](x, batch_edge_index, return_attention_weights=True)
            else:
                new_x = self.layers[i](x, batch_edge_index)

            # (2) Activation
            if self.use_activation:
                new_x = F.elu(new_x)
            
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
            
            # (5) Dropout
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 4. Dense Attention Map Transformation (Visualization)
        # 마지막 레이어에서 얻은 alpha를 Dense Matrix로 변환
        if 'alpha' in locals():
            node_batch = torch.arange(B, device=edge_idx_ret.device).repeat_interleave(C)  # shape (B*C,)
            # dense_attn: (B, C, C, heads)
            dense_attn = to_dense_adj(edge_idx_ret,         # (2, B*E), node index 0 ~ B*C-1
                                      batch=node_batch,     # (B*C,), 각 노드의 그래프 ID
                                      edge_attr=alpha,      # (B*E, heads)
                                      max_num_nodes=C       # 각 그래프당 노드 수
                                      )
            attn_map = dense_attn.mean(dim=-1)      # (B, C, C), 현재: [source, target]
            attn_map = attn_map.transpose(1, 2)     # (B, C, C), 변경 후: [target, source]
        # 5. Output Projection (Hidden -> L)
        # (B*C, Hidden) -> (B*C, L)
        out_flat = self.proj(x)
        
        # Restore Shape: (B, C, L) -> (B, L, C)
        out = out_flat.view(B, C, L).permute(0, 2, 1)

        return out, attn_map

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    '''
    단일 그래프 구조(org_edge_index)를 batch마다 offset을 두고 복사
    '''
    # org_edge_index: (2, E)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]

    # (2, E * B)
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous() 
        
    # Vectorized Offset Addition (for loop보다 빠름)
    # 0, 0, ..., C, C, ..., 2C, 2C ... 형태의 오프셋 벡터 생성
    offset = torch.arange(batch_num, device=org_edge_index.device).repeat_interleave(edge_num) * node_num
        
    # Source와 Target 모두에 오프셋 더하기
    batch_edge_index[0, :] += offset
    batch_edge_index[1, :] += offset

    return batch_edge_index.long()
