from tkinter.constants import HIDDEN
from typing import Any

import torch.nn as nn
import torch

import torch.nn.functional as F

"""
    B: 批量大小
    T: 序列长度
    E: 嵌入维度
"""


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim

        E, H, D = embed_dim, n_head, head_dim

        # (B,T,E) @ (E,E) --> (B,T,E)
        # QKV 三个线性变换的参数量：3E^2
        self.W_q = nn.Linear(E, H * D, bias=False)
        self.W_k = nn.Linear(E, H * D, bias=False)
        self.W_v = nn.Linear(E, H * D, bias=False)

        # 输出，参数量：E^2
        self.W_o = nn.Linear(H * D, E, bias=False)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        x: (B,T,E)
        """
        B, T, E = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)  # (B,T,E = H*D)  # 多个头的Q在一起呢
        K = self.W_k(x)  # (B,T,E = H*D)
        V = self.W_v(x)  # (B,T,E = H*D)

        Q = Q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        V = V.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        # 注意力分数
        # torch.matmul 矩阵乘法， QK转置
        # Q: (B,H,T,D)
        # K转置: (B,H,D,T)
        # QK转置: (B,H,T,T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / D ** 0.5  # 除以根号dk

        # 因果注意力掩码
        mask = torch.tril(torch.ones(T, T, device=scores.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 注意权重的计算
        weights = F.softmax(scores, dim=-1)  # (B,H,T,T)
        weights = self.attention_dropout(weights)
        hidden = torch.matmul(weights, V)  # (B,H,T,D)

        # (B,H,T,D) -> (B,T,H,D)     .contiguous 保证张量在内存中是连续排布的
        hidden = hidden.transpose(1, 2).contiguous()

        hidden = hidden.view(B, T, H * D)

        output = self.W_o(hidden)  # (B,T,E)
        output = self.output_dropout(output)

        return output


class FFN(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout_rate):
        super().__init__()

        # 共8E^2个参数
        self.layers = nn.Sequential(
            # (E, 4E) 共4E^2 个参数
            nn.Linear(embed_dim, ff_dim),  # (B,T,E) @ (E,4E) --> (B,T,4E)
            nn.GELU(),
            # (4E,E) 共4E^2 个参数
            # (B,T,4E) @ (4E, E) --> (B,T,E)
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, dropout_rate=0.1):
        super().__init__()
        # 头的维度
        # 词嵌入维度必须能够被头的数量整除
        head_dim = embed_dim // n_head

        # 第一次层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        # 多头注意力（计算token之间的相关性）
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)

        # 第二次层归一化
        self.norm2 = nn.LayerNorm(embed_dim)
        # 前馈网络（计算每个token在上下文中的含义）
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class GPT2(nn.Module):

    def __init__(
            self,
            vocab_size,
            max_context_len,
            embed_dim,
            n_head,
            n_layer,
            ff_dim,
            dropout_rate,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # 词嵌入的查找表
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 可学习的位置编码（不具备外推性）
        self.pos_embed = nn.Embedding(max_context_len, embed_dim)
        # 失活层
        self.dropout = nn.Dropout(dropout_rate)

        # 多个transformer block
        self.blocks = nn.ModuleList([Block(embed_dim, n_head, ff_dim, dropout_rate) for _ in range(n_layer)])

        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 输出层 (B,T,E) @ (E,V) --> (B,T,V)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # 初始化科学系参数
        self.apply(self._init_weights)

    def forward(self, ids):
        B, T = ids.shape
        device = ids.device

        # 位置的索引: [0,1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        # ids的词嵌入: (B,T,E)
        emb = self.embed(ids)
        # 位置编码: (T,E)
        pos_emb = self.pos_embed(pos)
        # pos_emb 会广播 x.shape:(B,T,E)
        x = self.dropout(emb + pos_emb)

        # 送入多层transformer block
        for block in self.blocks:
            x = block(x)  # x.shape:(B,T,E)

        # 最后的层归一化，输出是last_hidden_state: (B,T,E), 最后一个隐藏层
        x = self.norm(x)  # x.shape:(B,T,E)

        #  输出logits： logits.shape: (B,T,V)
        logits = self.lm_head(x)  # logits.shape:(B,T,V)

        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
