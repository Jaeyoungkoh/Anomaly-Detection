import torch
import torch.nn as nn
import torch.nn.functional as F

class GATv2Pytorch(nn.Module):
    """
    Paper-like GATv2 (dense fully-connected graph on nodes):

    For each head h:
      h_i^h = W^h x_i
      e_ij^h = a_h^T LeakyReLU( W_cat^h [h_i^h || h_j^h] )
      alpha_ij^h = softmax_j(e_ij^h)
      out_i^h = Σ_j alpha_ij^h h_j^h

    Then concat(or average) over heads.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super(GATv2Pytorch, self).__init__()

        self.in_channels = in_channels            # Hidden_dim  
        self.out_channels = out_channels          # per-head dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(negative_slope)

        H, Fin, D = heads, in_channels, out_channels

        # (1) Head-wise independent projection: W_proj[h] : Fin -> D
        self.W_proj = nn.Parameter(torch.empty(H, Fin, D))
        self.b_proj = nn.Parameter(torch.zeros(H, D)) if bias else None
        nn.init.xavier_uniform_(self.W_proj)

        # (2) Head-wise W_cat: (H, D, 2D) + bias (H, D)
        self.W_cat_weight = nn.Parameter(torch.empty(H, D, 2 * D))
        self.W_cat_bias = nn.Parameter(torch.zeros(H, D)) if bias else None
        self.a = nn.Parameter(torch.empty(H, D))
        nn.init.xavier_uniform_(self.W_cat_weight)
        nn.init.xavier_uniform_(self.a)

        # (4) Output bias (optional)
        if bias:
            if concat:
                self.bias_param = nn.Parameter(torch.zeros(H * D))
            else:
                self.bias_param = nn.Parameter(torch.zeros(D))
        else:
            self.register_parameter("bias_param", None)

    def forward(self, x, return_attention_weights: bool = False):
        """
        x: (B, N, H)
        """
        B, N, Fin = x.size()
        H, D = self.heads, self.out_channels
        assert Fin == self.in_channels

        # 1) head-wise projection
        h = torch.einsum("bni,hid->bhnd", x, self.W_proj)  # (B,H,N,D)
        if self.b_proj is not None:
            h = h + self.b_proj.view(1, H, 1, D)

        # 2) prepare concat pairs: (B, H, N, N, 2D)
        a_input = self._make_attention_input(h)

        # 3) head-wise W_cat and activation: z (B,H,N,N,D)
        #    z[b,h,i,j,d] = Σ_m W_cat_weight[h,d,m] * a_input[b,h,i,j,m] + bias[h,d]
        z = torch.einsum("bhijn,hdn->bhijd", a_input, self.W_cat_weight)  # (B,H,N,N,D)
        if self.W_cat_bias is not None:
            z = z + self.W_cat_bias.view(1, H, 1, 1, D)
        z = self.leakyrelu(z)

        # 4) attention logits: e (B,H,N,N)
        #    e = Σ_d z * a
        e = (z * self.a.view(1, H, 1, 1, D)).sum(dim=-1)

        # 5) softmax + dropout
        attention = torch.softmax(e, dim=-1)  # (B, H, N, N)
        attention_do = F.dropout(attention, self.dropout, training=self.training)

        # 6) aggregation: (B,H,N,N) @ (B,H,N,D) -> (B,H,N,D)
        out = torch.matmul(attention_do, h)

        # 7) concat or mean
        if self.concat:
            out = out.transpose(1, 2).reshape(B, N, H * D)   # (B,N,H*D)
        else:
            out = out.mean(dim=1)                            # (B,N,D)

        # 8) output bias
        if self.bias_param is not None:
            out = out + self.bias_param

        return out, attention.mean(dim=1) if return_attention_weights else out

    @staticmethod
    def _make_attention_input(v):
        """
        v: (B, H, N, D) -> (B, H, N, N, 2D)
        """
        B, H, N, D = v.size()

        rep = v.repeat_interleave(N, dim=2)      # (B, H, N*N, D)  i index repeats
        alt = v.repeat(1, 1, N, 1)               # (B, H, N*N, D)  j index cycles
        combined = torch.cat((rep, alt), dim=-1) # (B, H, N*N, 2D)

        return combined.view(B, H, N, N, 2 * D)