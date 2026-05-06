import torch
import torch.nn as nn

class LSTM_AE(nn.Module):
    def __init__(self, args):
        super(LSTM_AE, self).__init__()
        self.device = args.device
        self.nb_feature =args.input_c
        self.num_layers = 1
        self.hidden_size = 64
        self.dropout = 0

        self.encoder = Encoder(self.num_layers, self.hidden_size, self.nb_feature, self.dropout, self.device)
        self.decoder = Decoder(self.num_layers, self.hidden_size, self.nb_feature, self.dropout, self.device)

    def forward(self, input_seq):
        # Device 불일치 에러 수정
        output = torch.zeros(size=input_seq.shape, dtype=torch.float).to(self.device)
        
        # 1. 인코더 통과 (컨텍스트 벡터 생성)
        hidden_cell = self.encoder(input_seq)
        
        # 2. 디코더의 첫 입력값 설정 (시퀀스의 마지막 값)
        input_decoder = input_seq[:, -1, :].unsqueeze(1) # .view() 대신 직관적인 unsqueeze 사용
        
        # 3. 역순으로 디코딩 (논문 방식)
        for i in range(input_seq.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder # Autoregressive (이전 출력을 다음 입력으로)
            output[:, i, :] = output_decoder.squeeze(1)
            
        return output, None

class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, input_seq):
        # initHidden() 제거. PyTorch가 자동으로 (0, 0) 상태로 초기화하여 연산함
        _, hidden_cell = self.lstm(input_seq)
        return hidden_cell

class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)

    def forward(self, input_seq, hidden_cell):
        output, hidden_cell = self.lstm(input_seq, hidden_cell)
        output = self.linear(output)
        return output, hidden_cell