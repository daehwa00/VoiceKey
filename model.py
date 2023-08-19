import torch
from torch import nn
from typing import Type
import torch.nn.functional as F


def custom_chunking(tensor, chunk_size, step):
    chunks = []
    start = 0

    for _ in range(22):
        chunks.append(tensor[:, :, start : start + chunk_size])
        start += step
    return chunks


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ChunkedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ChunkedConvolution, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.swish = Swish()

    def forward(self, chunks):
        return [self.dropout(self.swish(self.bn(self.conv(chunk)))) for chunk in chunks]


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, feature):
        Q = self.query(feature)
        K = self.key(feature)
        V = self.value(feature)

        attention = F.softmax(Q @ K.transpose(1, 2), dim=-1)
        out = attention @ V
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(embed_size) for _ in range(num_heads)]
        )
        self.fc = nn.Linear(num_heads * embed_size, embed_size)

    def forward(self, feature):
        concat_attention = torch.cat([head(feature) for head in self.heads], dim=-1)
        out = self.fc(concat_attention)
        return out


class VoiceKeyModel(nn.Module):
    def __init__(self, dim: int = 64, num_heads: int = 8):
        super(VoiceKeyModel, self).__init__()

        self.conv1 = ChunkedConvolution(1, dim // 4, kernel_size=26)
        self.conv2 = ChunkedConvolution(dim // 4, dim // 2, kernel_size=26)
        self.conv3 = ChunkedConvolution(dim // 2, dim, kernel_size=25)

        self.attention = MultiHeadAttention(dim, num_heads)
        self.fc = nn.Linear(dim, dim)

    def forward(
        self, audio_vec1: torch.Tensor, audio_vec2: torch.Tensor
    ) -> torch.Tensor:
        chunks1 = custom_chunking(audio_vec1, 75, 25)
        chunks2 = custom_chunking(audio_vec2, 75, 25)

        # Extract features using convolutional layers
        chunks1 = self.conv1(chunks1)
        chunks2 = self.conv1(chunks2)

        chunks1 = self.conv2(chunks1)
        chunks2 = self.conv2(chunks2)

        chunks1 = self.conv3(chunks1)
        chunks2 = self.conv3(chunks2)

        # Concatenate the chunks
        feature1 = torch.cat(chunks1, dim=2).transpose(1, 2)
        feature2 = torch.cat(chunks2, dim=2).transpose(1, 2)

        # Apply attention mechanism
        feature1 = self.attention(feature1)  # (batch_size, seq_length, feature_dim)
        feature2 = self.attention(feature2)

        # Global average pooling
        feature1 = torch.mean(feature1, dim=1)
        feature2 = torch.mean(feature2, dim=1)

        # Pass through a fully connected layer
        feature1 = self.fc(feature1)
        feature2 = self.fc(feature2)

        return feature1, feature2
