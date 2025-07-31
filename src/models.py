# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation_fn(name):
    """Return the activation function corresponding to the name."""
    if name == 'silu':
        return F.silu
    elif name == 'gelu':
        return F.gelu
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class ResidualBlock(nn.Module):
    """A residual block for 1D convolutions with selectable activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2, activation_fn='silu'):
        super(ResidualBlock, self).__init__()
        self.activation = get_activation_fn(activation_fn)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.activation(out)
        return out

class CNNLSTM(nn.Module):
    """An improved CNN-LSTM model with selectable activation function."""
    def __init__(self, 
                 input_dim=768, 
                 num_classes=2, 
                 cnn_out_channels=128, 
                 lstm_hidden_dim=128, 
                 lstm_layers=2, 
                 dropout_rate=0.5,
                 activation_fn='silu'): # Add activation_fn as an argument
        
        super(CNNLSTM, self).__init__()

        self.res_block1 = ResidualBlock(input_dim, cnn_out_channels, activation_fn=activation_fn)
        self.res_block2 = ResidualBlock(cnn_out_channels, cnn_out_channels, activation_fn=activation_fn)
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        self.attention_pooling = AttentionPooling(input_dim=lstm_hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.res_block1(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.res_block2(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        
        # Attention Pooling
        attention_pooled = self.attention_pooling(lstm_out)
        
        x = self.dropout(attention_pooled)
        logits = self.fc(x)
        
        return logits

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
    def forward(self, lstm_out):
        attention_scores = self.attention_weights(lstm_out)
        attention_probs = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_probs, dim=1)
        return context_vector