import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes (k)
    :param window_size: length of the input sequence (n)
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x : (b,n,k)
        x = x.permute(0, 2, 1) # (b,k,n)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back

class CausalConvLayer(nn.Module):
    """1-D Causal Convolution layer.
    :param n_features: Number of input features/nodes (k)
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(CausalConvLayer, self).__init__()
        # Causal을 위해 왼쪽에만 (kernel_size - 1) 만큼의 패딩을 추가합니다.
        self.padding = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        
        # Conv1d 자체의 padding은 사용하지 않습니다.
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x : (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # (batch, seq_len, features)

class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes (k)
    :param window_size: length of the input sequence (n)
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True, individual=None):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features # k
        self.window_size = window_size # n
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias
        self.individual = individual

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        # 크기 (a_input_dim, 1)인 학습 가능한 파라미터 a 정의
        # 초기값은 비어 있음(empty)
        # model.parameters()에 포함되어 optimizer가 이 a 를 학습할 수 있게 됨

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1) # (b, k, n)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu

        # (1) e 계산하기
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*n)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1) > (b, k, k)

        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1) > (b, k, k)

        if self.use_bias:
            e += self.bias

        # (2) Attention weight 계산하기 ( softmax(e) )
        attention = torch.softmax(e, dim=2) # i(발신) 기준 j(수신)의 attention 구하기
        attention_do = torch.dropout(attention, self.dropout, train=self.training) # (b, k, k)

        # (3) 노드 업데이트 
        # x shape : (b, n, k)
        h = self.sigmoid(torch.matmul(attention_do, x)) # (b, k, n)
        out = h.permute(0, 2, 1) # (b, n, k)
        
        return out, attention

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """
        # v : (b, k, n)
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix (b, K*K, n)
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix (b, K*K, n)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*n)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)

# x > z
class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x): # (B, L, N*D)
        out, h = self.gru(x) # out : (B, L, H) / h: (Layer, B, H)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


# Reconstruction Model에서 사용될 Decoder
class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x) # (B, L, H)
        return decoder_out

class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()

        self.window_size = window_size

        # RNN Decoder
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x, recon_select):
        # x will be last hidden state of the GRU layer
        h_end = x # (B, H)

        if recon_select == 'RNN':

            h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1) # (B, L*H) > (B, L, H)
            decoder_out = self.decoder(h_end_rep) # (B, L, H)
            out = self.fc(decoder_out) # (B, L, D)
            
            return out # (B, L, D)


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)] 
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.individual = None

    def forward(self, x):
        for i in range(len(self.layers)-1): # len(self.layers) : 4
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)

        return self.layers[-1](x)
    
    
class Cluster_wise_linear(nn.Module): # 입력 시퀀스에 대해 군집 별 feed-forward 네트워크를 적용한 뒤, 군집 할당 확률을 사용해 각 군집의 결과를 가중합하여 출력하는 모듈 
    def __init__(self, n_cluster, n_vars, in_dim, out_dim):
        super().__init__()
        self.n_cluster = n_cluster
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linears = nn.ModuleList()
        for i in range(n_cluster):
            self.linears.append(nn.Linear(in_dim, out_dim))
        
    def forward(self, x, prob):
        # x: [bs, n_vars, in_dim]
        # prob: [n_vars, n_cluster]
        # return: [bs, n_vars, out_dim]
        x = x.unsqueeze(1).repeat(1, self.n_vars, 1)
        bsz = x.shape[0]
        output = []
        for layer in self.linears:
            output.append(layer(x))
        output = torch.stack(output, dim=-1).to(x.device)  #[bsz, n_vars, out_dim, n_cluster]
        prob = prob.unsqueeze(-1)  #[n_vars, n_cluster, 1]
        output = torch.matmul(output, prob).reshape(bsz, -1, self.out_dim)   #[bsz, n_vars, out_dim]

        return output
    


class AttentionFuser(nn.Module):
    """
    Attention-based Feature Fusion module.
    
    This module takes multiple feature streams, concatenates them, and then
    applies a self-attention mechanism to learn the dynamic inter-relationships
    among these features at each time step. It helps the model to weigh
    different features (like trend, seasonality, GAT outputs) adaptively.
    """
    def __init__(self, n_features, n_heads=4, dropout=0.1):
        super(AttentionFuser, self).__init__()
        
        # We have 5 feature streams to fuse.
        # DLinear(seasonal, trend), Conv1D, TemporalGAT, FeatureGAT
        self.embed_dim = n_features
        
        # Ensure that the embedding dimension is divisible by the number of heads.
        if self.embed_dim % n_heads != 0:
            raise ValueError(
                f"Embedding dimension ({self.embed_dim}) must be divisible by the number of heads ({n_heads})."
                f"Please adjust n_features or n_heads."
            )
            
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # Important: a batch dimension is first (b, n, k)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        
    def forward(self, x):
        # (1) Concatenate all feature streams along the feature dimension (dim=2)

        # (2) Apply self-attention
        # Query, Key, and Value are all the same concatenated tensor
        attn_output, _ = self.attention(x, x, x)
        
        # (3) Add & Norm (Residual connection)
        # This helps in stabilizing the training
        fused_features = self.norm(x + attn_output)
        
        return fused_features    
    

class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()

        # --- b : batch size
        # --- n : window size ( = backcast length )
        # --- k : num of featrues

        self.window_size = window_size

        # VAE Decoder
        self.fc_out_mean = nn.Linear(hid_dim, out_dim)
        self.fc_out_logvar = nn.Linear(hid_dim, out_dim)
        self.fc_mu = nn.Linear(hid_dim, hid_dim)
        self.fc_logvar = nn.Linear(hid_dim, hid_dim)

        # RNN Decoder
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)



    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x # (b, h)

        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1) # (b, h*n) > (b, n, h)
        decoder_out = self.decoder(h_end_rep) # (b, n, h)
        out = self.fc(decoder_out) # (b, n, k)
        
        return out # (b, n, k)



class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)] 
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.individual = None

    def forward(self, x):
        for i in range(len(self.layers)-1): # len(self.layers) : 4
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)

        return self.layers[-1](x)
    
class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x): # (b, n, 3*k)
        out, h = self.gru(x) # out : (b, n, h) / h: (L, b, h)
        return out, h[-1, :, :]   