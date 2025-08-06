# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation_fn(name):
    """
    Retrieve a PyTorch activation function by its string name.

    This helper function allows for the dynamic selection of activation
    functions within the model architecture based on hyperparameter choices.

    Args:
        name (str): The name of the activation function (e.g., 'silu', 'gelu').

    Returns:
        callable: The corresponding PyTorch activation function.
    """
    if name == 'silu':
        return F.silu
    elif name == 'gelu':
        return F.gelu
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class ResidualBlock(nn.Module):
    """
    Create a residual block with two 1D convolutional layers.

    This block implements a skip connection (residual connection), which allows
    the model to learn identity mappings and helps mitigate the vanishing
    gradient problem, enabling the training of deeper networks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        activation_fn (str, optional): Name of the activation function to use.
                                       Defaults to 'silu'.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2, activation_fn='silu'):
        super(ResidualBlock, self).__init__()
        self.activation = get_activation_fn(activation_fn)
        
        # Define the main path of the residual block.
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Define the shortcut (skip) connection.
        # If input and output dimensions differ, use a 1x1 convolution to match them.
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        """Define the forward pass of the residual block."""
        # Calculate the output of the main path.
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # Add the shortcut connection to the main path's output.
        out += self.shortcut(x)
        
        # Apply the final activation function.
        out = self.activation(out)
        return out

class AttentionPooling(nn.Module):
    """
    Create a self-attention pooling layer.

    This layer learns to compute a weighted average over a sequence, allowing
    the model to focus on the most informative time steps for the classification
    task, rather than treating all time steps equally (as in mean pooling).

    Args:
        input_dim (int): The feature dimension of the input sequence.
    """
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        # Use a linear layer to learn the attention scores for each time step.
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, lstm_out):
        """Define the forward pass of the attention mechanism."""
        # lstm_out shape: (batch_size, seq_len, input_dim)
        
        # Calculate a score for each time step.
        attention_scores = self.attention_weights(lstm_out)
        
        # Convert scores to probabilities that sum to 1 using softmax.
        attention_probs = F.softmax(attention_scores, dim=1)
        
        # Compute the weighted sum of the sequence using the attention probabilities.
        context_vector = torch.sum(lstm_out * attention_probs, dim=1)
        
        return context_vector

class CNNLSTM(nn.Module):
    """
    Build a Convolutional-LSTM model for sequence classification.

    This model processes sequences of embeddings by first applying 1D
    convolutional residual blocks to extract local hierarchical patterns,
    followed by a bidirectional LSTM to model long-term temporal dependencies,
    and a final attention layer to pool the sequence for classification.

    Args:
        input_dim (int, optional): The dimension of the input features for each time step.
                                   Defaults to 768 for Wav2Vec2-base.
        num_classes (int, optional): The number of output classes. Defaults to 2.
        cnn_out_channels (int, optional): Number of output channels from the CNN layers. Defaults to 128.
        lstm_hidden_dim (int, optional): Number of hidden units in the LSTM layer. Defaults to 128.
        lstm_layers (int, optional): Number of recurrent layers in the LSTM. Defaults to 2.
        dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.5.
        activation_fn (str, optional): Name of the activation function for CNN blocks.
                                       Defaults to 'silu'.
    """
    def __init__(self, 
                 input_dim=768, 
                 num_classes=2, 
                 cnn_out_channels=128, 
                 lstm_hidden_dim=128, 
                 lstm_layers=2, 
                 dropout_rate=0.5,
                 activation_fn='silu'):
        
        super(CNNLSTM, self).__init__()

        # Define the sequence of layers in the model, residual blocks for stable CNN feature extraction.
        self.res_block1 = ResidualBlock(input_dim, cnn_out_channels, activation_fn=activation_fn)
        self.res_block2 = ResidualBlock(cnn_out_channels, cnn_out_channels, activation_fn=activation_fn)
        
        # Bidirectional LSTM layer to capture long-term temporal dependencies.
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        # The input dimension to the attention layer is doubled due to the bidirectional LSTM.
        self.attention_pooling = AttentionPooling(input_dim=lstm_hidden_dim * 2)
        
        # Final classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            torch.Tensor: The final output logits of shape (batch_size, num_classes).
        """
        # Reshape for CNN layers, which expect (batch, channels, length).
        x = x.permute(0, 2, 1)

        # Pass through the convolutional blocks.
        x = self.res_block1(x)
        # Downsample the sequence length to reduce computation and create a more abstract representation.
        x = F.max_pool1d(x, kernel_size=2)
        x = self.res_block2(x)
        
        # Reshape for the LSTM layer, which expects (batch, sequence, features).
        x = x.permute(0, 2, 1)
        
        # Pass through the bidirectional LSTM.
        lstm_out, _ = self.lstm(x)
        
        # Use attention to pool the LSTM outputs into a single context vector.
        attention_pooled = self.attention_pooling(lstm_out)
        
        # Pass through the final classifier head.
        x = self.dropout(attention_pooled)
        logits = self.fc(x)
        
        return logits