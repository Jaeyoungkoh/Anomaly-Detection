import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import time
from torch_geometric.nn import GCNConv, GATConv, EdgeConv

import math

from layers.graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):

    '''
    단일 그래프 구조(org_edge_index)를 batch마다 offset을 두고 복사
    각 노드 인덱스를 i * node_num만큼 shift하여 각 batch가 독립적으로 그래프를 구성할 수 있게 합니다.
    '''
    # org_edge_index:(2, edge_num)
    #tensor([[ 1,  2,  3,  ..., 25, 26, 27],   # Source
    #        [ 0,  0,  0,  ..., 28, 28, 28]])  # Target
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]

    # .repeat(1,batch_num) : 0차원 1번 반복, 1차원 batch_num번 반복
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous() # torch.Size([2, edge_num*batch_num])

    # 노드번호를 배치별로 분리 (offset 적용)
    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1)) # 마지막 층 출력 차원 1
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x # (Batch_size, Node_num, Embed_dim)

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d): # 만약 현재 mod = BatchNorm1d라면 → 차원을 permute 해서 적용
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        '''
        x : (Batch_size*Node_num, Window)
        batch_gated_edge_index : (2, Node_num*topk*Batch_size)
        node_num : Batch_size*Node_num
        embedding : (Batch_size*Node_num, embed_dim)
        '''            
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True) 
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        # out : (Batch_size*Node_num, embed_dim)
        # new_edge_index : (2, Batch_size*Node_num)
        # att_weight : (Batch_size*Node_num)


        out = self.bn(out)
        
        return self.relu(out), self.att_weight_1, self.edge_index_1


class GDN(nn.Module):
    def __init__(self, args):

        super(GDN, self).__init__()

        self.edge_index_sets = args.edge_index_sets
        #[tensor([[ 1,  2,  3,  ..., 25, 26, 27],
        #       [ 0,  0,  0,  ..., 28, 28, 28]])]

        input_dim = args.win_size
        out_layer_num = args.out_layer_num
        out_layer_inter_dim = args.out_layer_inter_dim
        embed_dim = args.embed_dim
        node_num = args.input_c
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 0 ~ (node_num-1) 범위의 정수 입력이 들어오면, 각 정수에 대응되는 embed_dim 차원의 벡터를 출력
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(self.edge_index_sets) # 1
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, embed_dim, inter_dim=embed_dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = args.topk
        self.learned_graph = None

        self.out_layer = OutLayer(embed_dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):
        
        # data(input) : (Batch_size, Node_num, Window)

        x = data.clone().detach() # .detach() : 이후 x는 gradient를 추적하지 않음

        edge_index_sets = self.edge_index_sets

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous().to(self.device) # (Batch_size*Node_num, Window)

        '''
        현재 코드는 batch_edge_index 대신 batch_gated_edge_index 사용
        ㅁ batch_edge_index : 고정된 그래프 구조
        ㅁ batch_gated_edge_index : 노드 임베딩 간 코사인 유사도 Top-k
        '''

        # 각 edge_index에 대해 batch_edge_index 생성        
        gcn_outs = []

        for i, edge_index in enumerate(edge_index_sets): # i=0, edge_index_sets.shape : (2, N * N-1)
            edge_num = edge_index.shape[1] # N * N-1
            cache_edge_index = self.cache_edge_index_sets[i] # None

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(self.device) # (2, edge_num*Batch_num)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            # tensor([[  1,   2,   3,  ..., 924, 925, 926],
            #         [  0,   0,   0,  ..., 927, 927, 927]]

            '''
            ***************** Graph Structure Learning *****************
            학습 가능한 개별노드 임베딩 v_i을 통해 Adjacency Matrix A_ji (Top-K) 도출

            e_ji = cos_ji_mat (normalized dot product)
            '''

            # 개별 노드 임베딩 vi (nn.Embedding을 통해 학습 가능한 임베딩)
            # [0, 1, 2, ..., node_num-1] 형태의 인덱스 텐서를 생성 & 각 인덱스에 대해 임베딩 벡터를 가져옴
            all_embeddings = self.embedding(torch.arange(node_num).to(self.device)) # (Node_num, Emb_dim) 

            weights_arr = all_embeddings.detach().clone() # (Node_num, Emb_dim)

            # 배치 확장
            all_embeddings = all_embeddings.repeat(batch_num, 1) # (Batch_size*Node_num, Emb_dim)
            weights = weights_arr.view(node_num, -1) # (Node_num, Emb_dim)

            # 임베딩 벡터 사이 코사인 유사도 계산
            cos_ji_mat = torch.matmul(weights, weights.T) # (N, N) cos_ji_mat[i][j] = ⟨embedding_i, embedding_j⟩
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1)) # (N, N)
            # weights.norm(dim=-1).view(-1,1) : torch.Size([Node_num]) → torch.Size([Node_num, 1]
            # weights.norm(dim=-1).view(1,-1) : torch.Size([Node_num]) → torch.Size([1, Node_num]
            cos_ji_mat = cos_ji_mat / normed_mat # element-wise divide

            dim = weights.shape[-1]
            topk_num = self.topk

            # Top-K 유사한 노드 Index 추출
            topk_values_ji, topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1) # j가 i에 끼치는 영향력이므로 열방향 Top-k
            self.learned_graph = topk_indices_ji # (Node_num, topk)

            # Source node index (gated_i) 만들기
            # gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(self.device).unsqueeze(0) # (1, Node_num*topk)
            gated_i = torch.arange(0, node_num, device=self.device).view(-1, 1).repeat(1, topk_num).flatten().unsqueeze(0)
            # Target node index (gated_j) 만들기
            gated_j = topk_indices_ji.flatten().unsqueeze(0) # (1, Node_num*topk)

            # 노드 쌍
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0) # (2, Node_num*topk)

            # 노드번호를 배치별로 분리 (offset 적용)
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(self.device) # (2, Node_num*topk*Batch_size)
            
            '''
            ***************** GNN Layer *****************
            GNN Layer를 통해 모든 노드에 대한 Representation zi 추출
            
            <Input 정보>
            -. x : (Batch_size*Node_num, Window)
            -. batch_gated_edge_index : (2, Node_num*topk*Batch_size)
            -. node_num : Batch_size*Node_num
            -. embedding : (Batch_size*Node_num, embed_dim)
            '''            
            gcn_out, att_weight_out, edge_index_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
            gcn_outs.append(gcn_out) # (Batch_size*Node_num, Embed_dim)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1) # (Batch_size, Node_num, Embed_dim)

        indexes = torch.arange(0,node_num).to(self.device)

        # 노드 임베딩 v_i와 GNN Layer 출력값 Representation z_i Element-wise 곱
        out = torch.mul(x, self.embedding(indexes)) # (Batch_size, Node_num, Embed_dim)
        
        # self.embedding(indexes) : (Node_num, Embed_dim) → (1, Node_num, Embed_dim) → Broadcasting to (Batch_size, Node_num, Embed_dim)

        out = out.permute(0,2,1) # (Batch_size, Embed_dim, Node_num)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1) # (Batch_size, Node_num, Embed_dim)

        out = self.dp(out)
        out = self.out_layer(out) # mlp layer
        out = out.view(-1, node_num) # (Batch_size, Node_num)

        return out, att_weight_out, edge_index_out, topk_indices_ji, weights
        