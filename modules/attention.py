import torch
from torch import nn
import math
from typing import Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query_layer = nn.Conv1d(dim, dim, kernel_size=1)
        self.key_layer = nn.Conv1d(dim, dim, kernel_size=1)
        self.value_layer = nn.Conv1d(dim, dim, kernel_size=1)

        self.fc = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)  # LayerNorm after attention
        self.norm2 = nn.LayerNorm(dim)  # LayerNorm after FC

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        context_vector = self.norm1(context_vector + query)

        context_vector = self.fc(context_vector)

        context_vector = self.norm2(context_vector + query)

        return context_vector, attention_weights


class AttentionModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(AttentionModel, self).__init__()
        # Multi-head attention module
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        # Compute self-attention
        attn_output, _ = self.multihead_attn(query, key)
        # Apply dropout
        output = self.dropout(attn_output)

        return output
