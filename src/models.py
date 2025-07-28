# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A residual block for 1D convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
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
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.silu(out)
        return out

# Self-Attention Pooling Layer
class AttentionPooling(nn.Module):
    """
    A self-attention pooling layer that learns to weight and aggregate a sequence.
    """
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        # A simple linear layer to compute attention scores
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, input_dim)
        
        # Compute attention scores
        # (batch_size, seq_len, 1)
        attention_scores = self.attention_weights(lstm_out)
        
        # Apply softmax to get attention probabilities
        # (batch_size, seq_len, 1)
        attention_probs = F.softmax(attention_scores, dim=1)
        
        # Compute the weighted sum of the sequence vectors
        # torch.sum(lstm_out * attention_probs, dim=1)
        # (batch_size, input_dim)
        context_vector = torch.sum(lstm_out * attention_probs, dim=1)
        
        return context_vector

class CNNLSTM(nn.Module):
    """
    A CNN-LSTM model with Residual Blocks and Attention Pooling.
    """
    def __init__(self, 
                 input_dim=768, 
                 num_classes=2, 
                 cnn_out_channels=128, 
                 lstm_hidden_dim=128, 
                 lstm_layers=2, 
                 dropout_rate=0.5):
        
        super(CNNLSTM, self).__init__()

        self.res_block1 = ResidualBlock(input_dim, cnn_out_channels)
        self.res_block2 = ResidualBlock(cnn_out_channels, cnn_out_channels)
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        # Attention Pooling Layer
        # The input to the attention layer is the output of the bidirectional LSTM
        # which has a dimension of lstm_hidden_dim * 2
        self.attention_pooling = AttentionPooling(input_dim=lstm_hidden_dim * 2)

        self.dropout = nn.Dropout(dropout_rate)
        # The input to the final linear layer is now the output of the attention pooling
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.res_block1(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.res_block2(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        
        # removed: mean_pooled = torch.mean(lstm_out, dim=1)
        # Attention Pooling
        attention_pooled = self.attention_pooling(lstm_out)
        
        x = self.dropout(attention_pooled)
        logits = self.fc(x)
        
        return logits