from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear

from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class GATv2Conv(MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int, 
                 heads: int = 1, 
                 concat: bool = True,
                 negative_slope: float = 0.2, 
                 dropout: float = 0.,
                 add_self_loops: bool = True, 
                 bias: bool = True,
                 share_weights: bool = False
                 ):
        super(GATv2Conv, self).__init__(node_dim=0, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=True) # 원래 bias=bias임
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels, bias=True)
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, bias=True)
            self.lin_r = Linear(in_channels[1], heads * out_channels, bias=True)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, 
                x: Union[Tensor, OptPairTensor], 
                edge_index: Adj,
                size: Size = None, 
                return_attention_weights=None):

        H, C = self.heads, self.out_channels
        x_l: OptTensor = None
        x_r: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)

        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)
        
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, 
                x_j: Tensor, 
                x_i: Tensor,
                index: Tensor, 
                ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
    
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                            self.in_channels,
                                            self.out_channels, self.heads)
    
