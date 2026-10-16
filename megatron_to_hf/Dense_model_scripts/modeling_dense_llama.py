import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_dense_llama import DenseLlamaConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class SelfAttention(nn.Module):
    def __init__(self, config: DenseLlamaConfig):
        super().__init__()
        self.nh = config.num_attention_heads
        self.nk = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.nh
        H = config.hidden_size
        self.q_proj = nn.Linear(H, self.nh * self.head_dim, bias=False)
        self.k_proj = nn.Linear(H, self.nk * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, self.nk * self.head_dim, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, x, position_ids=None):
        B, T, H = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nk, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nk, self.head_dim).transpose(1, 2)

        rep = self.nh // self.nk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        cos, sin = self.rope(position_ids.max().item() + 1, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin, position_ids)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        p = torch.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(p, v).transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, H, I):
        super().__init__()
        self.w1 = nn.Linear(H, I, bias=False)
        self.w3 = nn.Linear(H, I, bias=False)
        self.w2 = nn.Linear(I, H, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config: DenseLlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x, position_ids=None):
        x = x + self.self_attn(self.input_layernorm(x), position_ids=position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DenseLlamaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DenseLlamaConfig
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(self, config: DenseLlamaConfig):
        super().__init__(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, position_ids=position_ids)
        x = self.model.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)