import torch
import torch.nn as nn

class LSTM_VAE(nn.Module):
    def __init__(self, args):
        super(LSTM_VAE, self).__init__()
        self.n_dim = args.input_c
        self.intermediate_dim = args.intermediate_dim
        # 잠재 벡터(z)의 차원
        self.z_dim = args.z_dim
        
        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------
        self.encoder_lstm = nn.LSTM(self.n_dim, self.intermediate_dim, batch_first=True)
        self.mu_layer = nn.Linear(self.intermediate_dim, self.z_dim)
        self.logvar_layer = nn.Linear(self.intermediate_dim, self.z_dim)
        
        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------
        # TF 코드에서는 두 번째 레이어를 곧바로 n_dim으로 출력했으나, 
        # PyTorch에서는 LSTM(intermediate_dim) -> Linear(n_dim)으로 빼는 것이 정석이자 성능이 더 좋습니다.
        self.decoder_lstm = nn.LSTM(self.z_dim, self.intermediate_dim, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.intermediate_dim, self.n_dim)

    def reparameterize(self, mu, logvar):
        # Reparameterization Trick: z = mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: (Batch_size, Sequence_length, Channels)
        batch_size, seq_len, _ = x.shape
        
        # --- Encode ---
        enc_out, (hn, cn) = self.encoder_lstm(x)
        h_last = hn[-1] # 마지막 시점의 hidden state (B, intermediate_dim)
        
        mu = self.mu_layer(h_last)
        logvar = self.logvar_layer(h_last)
        
        # --- Sample ---
        z = self.reparameterize(mu, logvar)
        
        # --- Decode ---
        # z를 시퀀스 길이만큼 복사: (B, z_dim) -> (B, L, z_dim)
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_out, _ = self.decoder_lstm(z_repeated)
        x_recon = self.output_layer(dec_out)
        
        # solver.py에서 호환되도록 형태를 맞춤. (출력값, VAE Loss용 잠재변수)
        return x_recon, (mu, logvar)