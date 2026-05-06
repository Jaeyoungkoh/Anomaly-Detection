import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_size)
        self.relu = nn.ReLU(True)
        
    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size/4))
        self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
        self.linear3 = nn.Linear(int(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w
    
class USAD(nn.Module):
    def __init__(self, args):
        super(USAD, self).__init__()

        self.hidden_dim = args.hidden_dim_usad        
        # USAD는 입력 윈도우 전체를 flatten 하여 사용
        # w_size = win_size (시퀀스 길이) * num_features (변수 개수)
        self.w_size = args.win_size * args.input_c

        # z_size = window_size * hidden_size
        self.z_size = args.win_size * self.hidden_dim
        
        self.encoder = Encoder(self.w_size, self.z_size)
        self.decoder1 = Decoder(self.z_size, self.w_size)
        self.decoder2 = Decoder(self.z_size, self.w_size)

    def forward(self, x):
        # x shape: (Batch_size, Sequence_length, Channels)
        batch_size, seq_len, num_features = x.shape
        x_flat = x.view(batch_size, -1) # Flatten
        
        z = self.encoder(x_flat)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        
        # solver.py 에서 loss 계산을 용이하게 하기 위해 원래 형태로 Reshape
        w1 = w1.view(batch_size, seq_len, num_features)
        w2 = w2.view(batch_size, seq_len, num_features)
        w3 = w3.view(batch_size, seq_len, num_features)
        
        return w1, w2, w3