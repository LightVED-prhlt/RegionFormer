from enum import Enum
from typing import Tuple, Optional, List, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from transformers import GPT2LMHeadModel


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, _ = y.shape
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale  # [b, n, m, h]
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        
        # [LEGRAD] We must access the attention map AFTER softmax but BEFORE it's consumed.
        attention = attention.softmax(dim=2)  # softmax over key positions m
        
        # [LEGRAD] Hook for gradients: If we are tracking gradients (training or LeGrad), retain them.
        if attention.requires_grad:
            attention.retain_grad()

        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention  # return attention for visualization/LeGrad


class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward(self, x, y=None, mask=None, return_attn: bool = False):
        att = None
        att_out, att = self.attn(self.norm1(x), y, mask)
        x = x + att_out
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, att
        return x, None


class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super().__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y=None, mask=None, return_attn: bool = False):
        attns: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:
                x, att = layer(x, y, return_attn=return_attn)
            elif self.enc_dec:
                x, att = layer(x, x, mask, return_attn=return_attn)
            else:
                x, att = layer(x, y, mask, return_attn=return_attn)
            if return_attn and att is not None:
                attns.append(att)  # each att: [b, n, m, h]
        if return_attn:
            return x, attns
        return x


class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super().__init__()
        self.clip_length = clip_length
        self.dim_embedding = dim_embedding
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x, return_attn: bool = False):
        """
        If return_attn=True, returns (out, attns) where:
          - out: [B, prefix_length, dim_embedding]
          - attns: List[num_layers] of attention tensors [B, N, M, H] over the concatenated sequence
                   N = M = clip_length + prefix_length
        """
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)  # [B, clip_length, dim_emb]
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)  # [B, prefix_len, dim_emb]
        concat = torch.cat((x, prefix), dim=1)  # [B, clip_len + prefix_len, dim_emb]
        if return_attn:
            full_out, attns = self.transformer(concat, return_attn=True)
        else:
            full_out = self.transformer(concat)
            attns = None
        out = full_out[:, self.clip_length:]  # keep only prefix positions for GPT2 prompt
        if return_attn:
            return out, attns
        return out


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.mapping_type = mapping_type
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
            self._clip_length_for_mapper = None
        else:
            assert clip_length is not None, "clip_length is required for Transformer mapper"
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers)
            self._clip_length_for_mapper = clip_length

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def project_with_attn(self, prefix: torch.Tensor, return_attn: bool = False):
        """
        Helper used at validation time to obtain both projected prefix embeddings and (optionally) attention maps.
        - For MLP mapping: returns (out, None).
        - For Transformer mapping: returns (out, attns).
        """
        if return_attn and isinstance(self.clip_project, TransformerMapper):
            out, attns = self.clip_project(prefix, return_attn=True)
            return out, attns, self._clip_length_for_mapper
        else:
            out = self.clip_project(prefix)
            return out, None, self._clip_length_for_mapper

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor,
                mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.gpt.eval()
        return self