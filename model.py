import torch
from torch import nn
from typing import Type
import torch.nn.functional as F


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

        # Define 1D convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, dim // 4, kernel_size=25, padding=12),
            nn.BatchNorm1d(dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(dim // 4, dim // 2, kernel_size=25, padding=12),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(dim // 2, dim, kernel_size=25, padding=12),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.attention = MultiHeadAttention(dim, num_heads)
        # self.attention = SelfAttention(dim)  # Adjusted the dimension
        self.fc = nn.Linear(dim, dim)

    def forward(
        self, audio_vec1: torch.Tensor, audio_vec2: torch.Tensor
    ) -> torch.Tensor:
        # Extract features using convolutional layers
        feature1 = self.conv_layers(audio_vec1).transpose(1, 2)[:, ::15, :]
        feature2 = self.conv_layers(audio_vec2).transpose(1, 2)[
            :, ::15, :
        ]  # (batch_size, seq_length, feature_dim)

        # Apply attention mechanism
        feature1 = self.attention(feature1)  # (batch_size, seq_length, feature_dim)
        feature2 = self.attention(feature2)

        # Global average pooling
        feature1 = torch.mean(feature1, dim=1)
        feature2 = torch.mean(feature2, dim=1)

        # Pass through a fully connected layer
        feature1 = self.fc(feature1)
        feature2 = self.fc(feature2)

        # Normalize the features
        feature1_normalized = F.normalize(feature1, p=2, dim=1)
        feature2_normalized = F.normalize(feature2, p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(
            feature1_normalized, feature2_normalized, dim=1
        )

        return cosine_sim


class VoiceKeyModelWithConv(nn.Module):
    def __init__(self, dim: int = 64, num_heads: int = 8, num_layers: int = 3):
        super(VoiceKeyModelWithConv, self).__init__()

        # Define 1D convolutional layer for initial feature extraction
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=25, padding=12),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # MultiHead Attention layers
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(dim, num_heads) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(dim, dim)

    def forward(
        self, audio_vec1: torch.Tensor, audio_vec2: torch.Tensor
    ) -> torch.Tensor:
        # Initial feature extraction using convolutional layer
        feature1 = self.conv_layer(audio_vec1).transpose(1, 2)[:, ::15, :]
        feature2 = self.conv_layer(audio_vec2).transpose(1, 2)[:, ::15, :]

        # Pass through multiple attention layers
        for attention in self.attention_layers:
            feature1 = attention(feature1)
            feature2 = attention(feature2)

        # Global average pooling
        feature1 = torch.mean(feature1, dim=1)
        feature2 = torch.mean(feature2, dim=1)

        # Pass through a fully connected layer
        feature1 = self.fc(feature1)
        feature2 = self.fc(feature2)

        # Normalize the features
        feature1_normalized = F.normalize(feature1, p=2, dim=1)
        feature2_normalized = F.normalize(feature2, p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(
            feature1_normalized, feature2_normalized, dim=1
        )

        return cosine_sim
