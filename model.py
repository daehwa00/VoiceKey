from modules import MultiHeadAttention, Conv1DNet
import torch
from torch import nn
from typing import Type


class VoiceKeyModel(nn.Module):
    def __init__(
        self,
        cnn: Type[Conv1DNet] = Conv1DNet,
        attention: Type[MultiHeadAttention] = MultiHeadAttention,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.cnn = cnn()
        self.attention_layers = nn.ModuleList(
            [attention(self.cnn.out_features, num_heads=4) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(
            self.cnn.out_features * 600, 2
        )  # 2 classes: same voice or different voice (this setting can use cross entropy loss)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio1: torch.Tensor, audio2: torch.Tensor) -> torch.Tensor:
        cnn_out1 = self.cnn(audio1)
        cnn_out2 = self.cnn(audio2)
        attention_out = cnn_out1
        for attention in self.attention_layers:
            attention_out, _ = attention(attention_out, cnn_out2)
            attention_out = self.dropout(attention_out)
        # Linearize the output of the last attention layer
        attention_out = attention_out.view(attention_out.shape[0], -1)
        out = self.fc(attention_out)
        return out
