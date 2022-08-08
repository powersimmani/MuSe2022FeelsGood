import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from config import ACTIVATION_FUNCTIONS
import torch, math

class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)
        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            h_n = rnn_enc[1][0] # (ND*NL, BS, dim)
            batch_size = x.shape[0]
            h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.d_out) # (NL, ND, BS, dim)
            last_layer = h_n[-1].permute(1,0,2) # (BS, ND, dim)
            x_out = last_layer.reshape(batch_size, self.n_directions * self.d_out) # (BS, ND*dim)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out


class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params

        self.inp = nn.Linear(params.d_in, params.d_rnn, bias=False)

        if params.rnn_n_layers > 0:
            self.rnn = RNN(params.d_rnn, params.d_rnn, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)

        d_rnn_out = params.d_rnn * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.d_rnn
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)

        if self.params.rnn_n_layers > 0:
            x = self.rnn(x, x_len)
        y = self.out(x)
        return self.final_activation(y)

    def set_n_to_1(self, n_to_1):
        self.rnn.n_to_1 = n_to_1


class TFModel(nn.Module):
    def __init__(self, params):
        super(TFModel, self).__init__()
        self.params = params
        self.device = torch.device("cuda")
        d_model = params.d_rnn
        d_rnn_out = params.d_rnn * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.d_rnn

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=params.rnn_n_layers, dropout=params.rnn_dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=int(params.rnn_n_layers/2) )
        self.pos_encoder = PositionalEncoding(params.d_rnn, params.rnn_dropout)

        self.encoder = nn.Sequential(
            nn.Linear(params.d_in, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, src, src_len):
        srcmask = self.generate_square_subsequent_mask(src.shape[1]).to(self.device)
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.out(output)
        output =  self.final_activation(output)
    
        return output

    def set_n_to_1(self, n_to_1):#Just for corresponding with RNN
        self.n_to_1 = n_to_1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask  


