# style_encoder.py
import torch
import torch.nn as nn


class StyleEncoderConfig:
    def __init__(self, vocab_size: int, latent_dim: int = 32, hidden_dim: int = 256, n_layers: int = 2, n_heads: int = 4):
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads


class StyleEncoder(nn.Module):
    """
    小型 Transformer Encoder，将 (history + query) 的 token 序列
    映射成一个风格向量 zuq ∈ R^latent_dim。
    """
    def __init__(self, config: StyleEncoderConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # 池化 + 投影到 latent_dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L]
        attention_mask: [B, L]，1 表示有效 token，0 表示 pad（可以为 None）
        """
        x = self.embedding(input_ids)  # [B, L, H]

        # 处理 attention mask（TransformerEncoder 使用的是 key_padding_mask）
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True 表示需要 mask

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, H]

        # 全局平均池化
        x = x.transpose(1, 2)   # [B, H, L]
        x = self.pool(x).squeeze(-1)  # [B, H]

        zuq = self.proj(x)  # [B, latent_dim]
        return zuq
