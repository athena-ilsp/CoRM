import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_standard_moe import StandardMoEConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin, position_ids):
    cos = cos.index_select(0, position_ids.reshape(-1)).view(position_ids.size(0), position_ids.size(1), -1)
    sin = sin.index_select(0, position_ids.reshape(-1)).view(position_ids.size(0), position_ids.size(1), -1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cache(max_seq_len)

    def _set_cache(self, seq_len: int):
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, device, dtype):
        if seq_len > self.cos_cached.size(0):
            new_len = 1
            while new_len < seq_len:
                new_len *= 2
            self._set_cache(new_len)
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


class StandardRouter(nn.Module):
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, x: torch.Tensor):
        # x: [B, T, H]
        router_logits = self.gate(x)  # [B, T, E]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # DON'T renormalize - keep original softmax probabilities
        return routing_weights, selected_experts, router_logits


class ReLURouter(nn.Module):
    """ReLU router: all experts with positive logit receive the token."""
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.num_experts = config.num_local_experts

    def forward(self, x: torch.Tensor):
        router_logits = self.gate(x)           # [B, T, E]
        routing_weights = F.relu(router_logits).to(torch.float32)
        return routing_weights, router_logits


class ReLUMoE(nn.Module):
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        self.router = ReLURouter(config)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_local_experts)
        ])
        self.num_experts = config.num_local_experts

    def forward(self, x):
        routing_weights, _ = self.router(x)   # [B, T, E]
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            w = routing_weights[:, :, e]       # [B, T]
            mask = w > 0
            if mask.any():
                xe = x[mask]
                ye = expert(xe)
                y[mask] += ye * w[mask].unsqueeze(-1).to(ye.dtype)
        return y


class SwiGLUExpert(nn.Module):
    def __init__(self, H, I):
        super().__init__()
        self.w1 = nn.Linear(H, I, bias=False)  # gate_proj
        self.w3 = nn.Linear(H, I, bias=False)  # up_proj
        self.w2 = nn.Linear(I, H, bias=False)  # down_proj

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StandardMoE(nn.Module):
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        self.router = StandardRouter(config)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_local_experts)
        ])
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        shared_I = getattr(config, "shared_expert_intermediate_size", 0)
        if shared_I > 0:
            self.shared_expert = SwiGLUExpert(config.hidden_size, shared_I)
        else:
            self.shared_expert = None

    def forward(self, x):
        B, T, H = x.shape
        routing_weights, selected_experts, router_logits = self.router(x)
        # routing_weights: [B, T, top_k], selected_experts: [B, T, top_k]

        y = torch.zeros_like(x)

        # Loop over top-k selections
        for k_idx in range(self.top_k):
            expert_indices = selected_experts[:, :, k_idx]  # [B, T]
            weights = routing_weights[:, :, k_idx]  # [B, T]

            for e, expert in enumerate(self.experts):
                mask = (expert_indices == e)  # [B, T]
                if mask.any():
                    xe = x[mask]  # [num_selected, H]
                    ye = expert(xe)  # [num_selected, H]
                    y[mask] += ye * weights[mask].unsqueeze(-1).to(ye.dtype)

        # Shared expert: applied to all tokens, added unconditionally
        if self.shared_expert is not None:
            y = y + self.shared_expert(x)

        return y


class SelfAttention(nn.Module):
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        H = config.hidden_size
        nh = config.num_attention_heads
        nk = config.num_key_value_heads
        self.head_dim = H // nh
        self.nh = nh
        self.nk = nk

        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, nk * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, nk * self.head_dim, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

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
        out = torch.matmul(p, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.block_sparse_moe = StandardMoE(config)

    def forward(self, x, position_ids=None):
        x = x + self.self_attn(self.input_layernorm(x), position_ids=position_ids)
        x = x + self.block_sparse_moe(self.post_attention_layernorm(x))
        return x


class StandardMoEForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = StandardMoEConfig
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(self, config: StandardMoEConfig):
        super().__init__(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        **kwargs,
    ):
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


class XMoERouter(nn.Module):
    """X-MoE router (Chi et al., 2022): cosine routing with dim reduction + L2 norm + learnable temperature.

    Weights: proj_weight [de, H], expert_embeddings [E, de], temperature [1].
    Expert embeddings pinned at L2 norm 0.1 in training; at inference we just normalize.
    """
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        H = config.hidden_size
        E = config.num_local_experts
        de = getattr(config, 'xmoe_expert_dim', E // 2)

        self.num_experts = E
        self.top_k = config.num_experts_per_tok
        self.proj_weight = nn.Parameter(torch.empty(de, H))
        self.expert_embeddings = nn.Parameter(torch.empty(E, de))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):
        # x: [B, T, H]
        flat = x.view(-1, x.shape[-1])
        projected = F.normalize(F.linear(flat, self.proj_weight), dim=-1)
        embeddings = F.normalize(self.expert_embeddings, dim=-1)
        scores = projected @ embeddings.t()
        scores = scores / self.temperature.abs().clamp(min=1e-6)
        scores = scores.view(*x.shape[:-1], self.num_experts)

        routing_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        return routing_weights, selected_experts, scores


class XMoE(nn.Module):
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        self.router = XMoERouter(config)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_local_experts)
        ])
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, x):
        B, T, H = x.shape
        routing_weights, selected_experts, _ = self.router(x)
        flat_x = x.view(-1, H)
        flat_w = routing_weights.view(-1, self.top_k)
        flat_se = selected_experts.view(-1, self.top_k)

        y = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            mask = (flat_se == e).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            xe = flat_x[idx]
            ye = self.experts[e](xe)
            we = (flat_se[idx] == e).float() * flat_w[idx]
            we = we.sum(dim=-1, keepdim=True).to(ye.dtype)
            y[idx] += ye * we
        return y.view(B, T, H)


class XMoEBlock(nn.Module):
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.block_sparse_moe = XMoE(config)

    def forward(self, x, position_ids=None):
        x = x + self.self_attn(self.input_layernorm(x), position_ids=position_ids)
        x = x + self.block_sparse_moe(self.post_attention_layernorm(x))
        return x


class XMoEForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = StandardMoEConfig
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(self, config: StandardMoEConfig):
        super().__init__(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([XMoEBlock(config) for _ in range(config.num_hidden_layers)])
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


class ReLUBlock(nn.Module):
    def __init__(self, config: StandardMoEConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.block_sparse_moe = ReLUMoE(config)

    def forward(self, x, position_ids=None):
        x = x + self.self_attn(self.input_layernorm(x), position_ids=position_ids)
        x = x + self.block_sparse_moe(self.post_attention_layernorm(x))
        return x


class ReLUMoEForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = StandardMoEConfig
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(self, config: StandardMoEConfig):
        super().__init__(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([ReLUBlock(config) for _ in range(config.num_hidden_layers)])
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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