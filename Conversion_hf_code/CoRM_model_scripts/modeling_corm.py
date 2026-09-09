import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_corm import CoRMConfig


# -------------------------
# Small building blocks
# -------------------------
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

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len, device, dtype):
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


# -------------------------
# CoRM Router
# -------------------------
class CoRM(nn.Module):
    """
    CoRM: Contrastive Routing Mixture-of-Experts router.

    Routing logit for expert e:
        gap[e] = [Q_e(x) - Q_e(null)] · K(x) / sqrt(d2)

    Weights:
        key:              [d2, H]    shared key projection (no bias)
        expert_queries:   [E*d2, H]  per-expert query projection (no bias)
        global_null_mean: buffer [H] EMA of LN'd hidden states (not saved in DCP → zeros)
        routing_temperature: [1]
    """

    def __init__(self, config: CoRMConfig):
        super().__init__()
        H = config.hidden_size
        E = config.num_local_experts
        d2 = config.expert_attn_size

        self.num_experts = E
        self.expert_attn_size = d2
        self.top_k = config.num_experts_per_tok

        self.key = nn.Linear(H, d2, bias=False)
        self.expert_queries = nn.Linear(H, E * d2, bias=False)

        # EMA null state — zero-initialised; updated during warmup (train mode)
        self.register_buffer('global_null_mean', torch.zeros(H))
        self.register_buffer('ema_initialized', torch.tensor(False))
        self.momentum = getattr(config, 'router_background_momentum', 0.01)
        self.use_l2_normalize = getattr(config, 'moe_router_use_l2_normalize', True)

        self.routing_temperature = nn.Parameter(
            torch.tensor(config.routing_temperature_init)
        )

    def gating_logits(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        N = B * T
        E = self.num_experts
        d2 = self.expert_attn_size

        real = x.reshape(N, H)
        real_n = F.layer_norm(real, (H,), eps=1e-5)

        # Update EMA in train mode (mirrors Megatron router exactly)
        if self.training:
            with torch.no_grad():
                batch_mean = real_n.mean(dim=0).detach()
                if not self.ema_initialized.item():
                    self.global_null_mean.copy_(batch_mean)
                    self.ema_initialized.fill_(True)
                else:
                    self.global_null_mean.mul_(1 - self.momentum).add_(
                        self.momentum * batch_mean
                    )

        K = self.key(real_n)                                        # [N, d2]
        if self.use_l2_normalize:
            K = F.normalize(K, p=2, dim=-1)

        Q_real = self.expert_queries(real_n).view(N, E, d2)         # [N, E, d2]
        if self.use_l2_normalize:
            Q_real = F.normalize(Q_real, p=2, dim=-1)

        # Null queries: EMA is already in post-LN space, don't re-layernorm
        Q_null = self.expert_queries(self.global_null_mean).view(E, d2)  # [E, d2]
        if self.use_l2_normalize:
            Q_null = F.normalize(Q_null, p=2, dim=-1)

        scale = 1.0 / math.sqrt(d2)
        attn_real = (Q_real * K.unsqueeze(1)).sum(-1) * scale        # [N, E]
        attn_null = (Q_null.unsqueeze(0) * K.unsqueeze(1)).sum(-1) * scale  # [N, E]

        gap = attn_real - attn_null

        temp = torch.clamp(self.routing_temperature, min=0.1, max=10.0)
        router_logits = torch.clamp(gap * temp, min=-10.0, max=10.0)

        return router_logits.view(B, T, E)

    def topk_routing(self, logits: torch.Tensor):
        if self.top_k == 1:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
        else:
            topk_logits, selected_experts = torch.topk(logits, self.top_k, dim=-1)
            routing_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
        return selected_experts, routing_weights


# Only "corm" is supported — the only router type ever exported through the
# HF conversion pipeline.
ROUTER_REGISTRY = {
    "corm": CoRM,
}


def get_router_class(router_type: str) -> Type[nn.Module]:
    """
    Get router class by type string.

    Args:
        router_type: Must be "corm".

    Returns:
        Router class

    Raises:
        ValueError: If router_type is not recognized
    """
    if router_type not in ROUTER_REGISTRY:
        available = ", ".join(ROUTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown router type '{router_type}'. "
            f"Available types: {available}"
        )
    return ROUTER_REGISTRY[router_type]

# -------------------------
# Expert MLP (Mixtral-like)
# -------------------------
class SwiGLUExpert(nn.Module):
    def __init__(self, H, I):
        super().__init__()
        self.w1 = nn.Linear(H, I, bias=False)  # gate_proj
        self.w3 = nn.Linear(H, I, bias=False)  # up_proj
        self.w2 = nn.Linear(I, H, bias=False)  # down_proj

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CoRMMoE(nn.Module):
    """MoE layer using the CoRM router (no shared expert)."""

    def __init__(self, config: CoRMConfig):
        super().__init__()

        router_class = get_router_class(config.router_type)
        self.router = router_class(config)

        H = config.hidden_size
        I = config.intermediate_size
        E = config.num_local_experts
        self.experts = nn.ModuleList([SwiGLUExpert(H, I) for _ in range(E)])
        self.top_k = config.num_experts_per_tok

    def forward(self, x):
        B, T, H = x.shape
        logits = self.router.gating_logits(x)
        selected_experts, routing_weights = self.router.topk_routing(logits)

        y = torch.zeros_like(x)

        for k_idx in range(self.top_k):
            expert_indices = selected_experts[:, :, k_idx]
            weights = routing_weights[:, :, k_idx]

            for e, expert in enumerate(self.experts):
                mask = (expert_indices == e)
                if mask.any():
                    xe = x[mask]
                    ye = expert(xe)
                    y[mask] += ye * weights[mask].unsqueeze(-1).to(ye.dtype)

        return y


# -------------------------
# Transformer layer
# -------------------------
class SelfAttention(nn.Module):
    def __init__(self, config: CoRMConfig):
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

    def forward(self, x, attn_mask=None):
        B, T, H = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nk, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nk, self.head_dim).transpose(1, 2)

        # repeat kv to match query heads
        rep = self.nh // self.nk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        cos, sin = self.rope(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

        p = torch.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(p, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, config: CoRMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = CoRMMoE(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.moe(self.post_attention_layernorm(x))
        return x


class CoRMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CoRMConfig
    main_input_name = "input_ids"

    def __init__(self, config: CoRMConfig):
        super().__init__(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        x = self.model.embed_tokens(input_ids)

        for layer in self.model.layers:
            x = layer(x)

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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )
