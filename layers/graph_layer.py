import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import time
import math

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs) # aggr='add' → 이웃 노드의 메시지를 sum 방식으로 집계함

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # 가중치 초기화
    def reset_parameters(self):
        # Xavier 방식으로 가중치 초기화
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        
        # 0으로 초기화
        zeros(self.att_em_i)
        zeros(self.att_em_j)

        if self.bias is not None:
            zeros(self.bias)



    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        '''
        x : (Batch_size*Node_num, Window)
        batch_gated_edge_index : (2, Node_num*topk*Batch_size)
        embedding : (Batch_size*Node_num, embed_dim)
        '''       

        # Wx (Input embedding Linear trasform)
        if torch.is_tensor(x):
            x = self.lin(x)    # (Batch_size*Node_num, Embed_dim)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        # 기존에 있던 self-loop 제거
        edge_index, _ = remove_self_loops(edge_index)
  
        # 모든 노드에 대해 self-loop를 딱 하나씩 명시적으로 다시 추가 
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim)) # edge_index : (2, Node_num*topk*Batch_size)

        # 각 엣지 기준 메시지 생성(message) → 노드 기준 메세지 수신(aggregate) → 최종 노드 임베딩 업데이트(update)
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index, # out : (Batch_size*Node_num, Embed_dim)
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            if out.dim() == 3:
                out = out.mean(dim=1)
            elif out.dim() == 2:
                pass
            else:
                raise ValueError(f"Unexpected out shape before bias: {out.shape}")


        if self.bias is not None:
            if out.shape[-1] != self.bias.shape[0]:
                raise ValueError(f"Bias shape mismatch: out={out.shape}, bias={self.bias.shape}")
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, 
                x_i,                        # 수신노드 입력 Feature (Node_num*topk*Batch_size, Embed_dim)
                x_j,                        # 발신노드 입력 Feature (Node_num*topk*Batch_size, Embed_dim)
                edge_index_i,               # 수신 노드 인덱스 (Node_num*topk*Batch_size)
                size_i,                     # 수신 노드 개수 (Node_num*Batch_size)
                embedding,                  # 각 노드 추가 임베딩 정보 (Node_num*Batch_size, Embed_dim)
                edges,                      # 엣지 인덱스 (2, Node_num*topk*Batch_size)
                return_attention_weights):

        '''
        message()에서 각 엣지마다 attention-weighted 메시지 생성
        노드 input feature 와 노드 embedding 정보를 합쳐(Concat) attention 계산
        '''
        # 이전에 Linear를 통과한 입력값(사실 Wx_i, Wx_j)
        x_i = x_i.view(-1, self.heads, self.out_channels) # (Node_num*topk*Batch_size, 1, Embed_dim)
        x_j = x_j.view(-1, self.heads, self.out_channels) # (Node_num*topk*Batch_size, 1, Embed_dim)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1) # (Node_num*topk*Batch_size, 1, Embed_dim)
            embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1) # (Node_num*topk*Batch_size, 1, Embed_dim)
            
            # Concat Transformed input feature(Wx) & Sensor embedding(v) > (g_i, g_j), 노드의 전체 정보
            key_i = torch.cat((x_i, embedding_i), dim=-1) # (Node_num*topk*Batch_size, 1, Embed_dim*2)
            key_j = torch.cat((x_j, embedding_j), dim=-1) # (Node_num*topk*Batch_size, 1, Embed_dim*2)

        # a : a vector of learned coefficients for the attention mechanism
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1) # a_i (1,1,Embed_dim*2)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1) # a_j (1,1,Embed_dim*2)

        # attention score alpha : (g_i · a_i) + (g_j · a_j) 내적 합
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1) # (Node_num*topk*Batch_size, 1)
        
        alpha = alpha.view(-1, self.heads, 1) # (Node_num*topk*Batch_size, 1, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope) # (Node_num*topk*Batch_size, 1, 1)
        alpha = alpha.squeeze(-1).squeeze(-1) # (Node_num*topk*Batch_size)
        size_i = torch.tensor(size_i)

        # Softmax (edge_index_i 수신 노드별로 정규화)
        alpha = softmax(alpha, edge_index_i, size_i)
           
        if return_attention_weights:
            self.__alpha__ = alpha # (Node_num × topk × Batch_size)
        
        # Attention weight에 Dropout 적용
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # message shape correction (flatten to 2D) ← aggregate input으로 전달
        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.out_channels) # (Node_num*topk*Batch_size, head x embed_dim)

        # 기존 코드
        # return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
