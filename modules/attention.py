import torch
from torch import nn
import math


class DepthwiseConv1d(nn.Conv1d):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query_layer = DepthwiseConv1d(dim, kernel_size=1)
        self.key_layer = DepthwiseConv1d(dim, kernel_size=1)
        self.value_layer = DepthwiseConv1d(dim, kernel_size=1)

        self.fc_out = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        batch_size = query.shape[0]

        query = self.query_layer(query.permute(0, 2, 1)).permute(0, 2, 1)
        key = self.key_layer(key_value.permute(0, 2, 1)).permute(0, 2, 1)
        value = self.value_layer(key_value.permute(0, 2, 1)).permute(0, 2, 1)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        context_vector = torch.matmul(attention_weights, value).permute(0, 2, 1, 3)

        context_vector = context_vector.contiguous().view(batch_size, -1, self.dim)
        context_vector = self.fc_out(context_vector)

        context_vector = self.norm(context_vector)

        return context_vector, attention_weights


class AttentionModel(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(AttentionModel, self).__init__()
        # Multi-head attention module
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key):
        # Compute self-attention
        attn_output, _ = self.multihead_attn(query, key, key)
        # Apply dropout
        attn_output = self.dropout(attn_output)
        # Add residual connection and apply layer normalization
        output = self.norm(attn_output + query)
        return output
