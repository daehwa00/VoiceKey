import torch
from torch import nn
import math
from typing import Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_length = 600

        self.query_layer = nn.Conv1d(dim, dim, kernel_size=1)
        self.key_layer = nn.Conv1d(dim, dim, kernel_size=1)
        self.value_layer = nn.Conv1d(dim, dim, kernel_size=1)

        self.fc = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)  # LayerNorm after attention
        self.norm2 = nn.LayerNorm(dim)  # LayerNorm after FC

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]

        query = self.query_layer(query)  # [batch_size, 64, seq_length]
        key = self.key_layer(key_value)  # [batch_size, 64, seq_length]
        value = self.value_layer(key_value)  # [batch_size, 64, seq_length]

        query = query.view(
            batch_size, self.seq_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = key.view(
            batch_size, self.seq_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        value = value.view(
            batch_size, self.seq_length, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # [batch_size, num_heads, seq_length, head_dim]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [batch_size, num_heads, seq_length, seq_length]

        attention_weights = torch.softmax(scores, dim=-1)
        context_vector = torch.matmul(attention_weights, value).permute(
            0, 2, 1, 3
        )  # [batch_size, seq_length, num_heads, head_dim]

        context_vector = context_vector.contiguous().view(
            batch_size, -1, self.dim
        )  # [batch_size, seq_length, dim]

        query = query.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.dim)
        context_vector = self.norm1(context_vector + query)
        context_vector_fc = self.fc(context_vector)

        context_vector = self.norm2(context_vector + context_vector_fc)

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
