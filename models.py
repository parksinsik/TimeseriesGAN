import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class LSTMGenerator(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=1):
        
        super().__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Tanh())
        
        
    def forward(self, inputs):
        
        batch_size, seq_length = inputs.size(0), inputs.size(1)
        
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(inputs, (h_0, c_0))
        
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_length, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_length, self.output_dim)
        
        return outputs

    
class Chomp1d(nn.Module): # K
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
        
    def init_weights(self):
        
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    
    def __init__(self, n_inputs, n_channels, kernel_size=2, dropout=0.3):
        
        super(TemporalConvNet, self).__init__()
        
        layers = []
        n_levels = len(n_channels)
        
        for i in range(n_levels):
            
            dilation_size = 2 ** i
            
            in_channels = n_inputs if i == 0 else n_channels[i-1]
            out_channels = n_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1)*dilation_size, dropout=dropout)]
 
        self.net = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.net(x)


class TCN(nn.Module):
    
    def __init__(self, input_size, output_size, n_channels, kernel_size, dropout):
        
        super(TCN, self).__init__()
        
        self.tcn = TemporalConvNet(input_size, n_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(n_channels[-1], output_size)
        self.init_weights()
        
        
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        
    
    def forward(self, x, channel_last=True): #If channel_last, the expected format is (batch_size, seq_len, features)
        y = self.tcn(x.transpose(1, 2) if channel_last else x)
        return self.linear(y.transpose(1, 2))


class CausalConvDiscriminator(nn.Module):
    
    def __init__(self, input_size, n_layers, n_channel, kernel_size, dropout=0.3):        
        super().__init__()
        n_channels = [n_channel] * n_layers
        self.tcn = TCN(input_size, 1, n_channels, kernel_size, dropout)
                
    def forward(self, x, channel_last=True):
        return torch.sigmoid(self.tcn(x, channel_last))
