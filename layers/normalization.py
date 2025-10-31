import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Revin
입력 데이터의 분포를 인스턴스 단위로 정규화하고, 필요 시 원래의 스케일로 복원할 수 있도록 하는 인스턴스 정규화 모듈 
'''
class RevIN(nn.Module): 
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str): # 입력 텐서 x에 대해 정규화 또는 역정규화를 수행 
        '''
        x : (b,n,k)
        --- b : batch size
        --- n : window size
        --- k : num of featrues
        '''
        if mode == 'n':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'd':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self): # 학습 가능한 파라미터를 초기화 
        # initialize RevIN params: (K,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x): # 입력 데이터 x의 통계 정보를 계산 
        dim2reduce = tuple(range(1, x.ndim-1)) # (1,) 즉, Window dimension
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x): # 입력 데이터 x를 정규화 
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x): # 정규화된 데이터를 원래 스케일로 복원
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class DishTS(nn.Module):
    def __init__(self, n_series, seq_len, dish_init='uniform'):
        super().__init__()

        init = dish_init #'standard', 'avg' or 'uniform'
        activate = True
        n_series = n_series # number of series
        lookback = seq_len # lookback length
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2)/lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback+torch.rand(n_series, lookback, 2)/lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate


    def normalize(self, batch_x, dec_inp=None):
        # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
        # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
        self.preget(batch_x)
        batch_x = self.forward_process(batch_x)
        dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
        return batch_x, dec_inp

    def denormalize(self, batch_x):
        # batch_x: B*H*D (forecasts)
        batch_y = self.inverse_process(batch_x)
        return batch_y


    def preget(self, batch_x):
        # (B, T, N)
        x_transpose = batch_x.permute(2,0,1)   # (N, B, T)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:] 
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)

    def forward_process(self, batch_input):
        #print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst
    
    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih
    
    
    def forward(self, batch_x, mode='n', dec=None):
        if mode == 'n':
            return self.normalize(batch_x, dec)
        elif mode =='d':
            return self.denormalize(batch_x)
            