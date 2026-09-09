# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from megatron.core.transformer.moe.logging import setup_moe_logging, update_moe_routing_and_log, get_moe_stats_tracker
from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        logits = torch.nn.functional.linear(input, self.weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number

class TopKRouter(Router):
    """Route each token to the top-k experts with optional top-p filtering."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None
        
        # NEW: Dynamic routing parameters
        self.use_dynamic_routing = getattr(config, 'use_dynamic_routing', False)
        self.top_p = getattr(config, 'top_p', 1.0)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor."""
        # ... keep existing implementation ...
        

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            indices (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )

        if self.training:
            # Apply load balancing loss
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)
        return probs, routing_map

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,):
        """Applies auxiliary loss to the MoE layer.

        Args:
            probs (torch.Tensor): The probs output by the router for each token.
                [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert.
                [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = switch_load_balancing_loss_func(
            probs,
            num_local_tokens_per_expert,
            self.topk,
            moe_aux_loss_coeff,
            sequence_partition_group=sequence_partition_group,
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function with optional top-p filtering

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        # ------------------------------------------------------------------
        # NEW: OPTIONAL TOP-P FILTERING (applied before routing)
        # ------------------------------------------------------------------
        if self.use_dynamic_routing and self.top_p < 1.0:
            # Compute probabilities for top-p selection
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            
            # Sort probabilities in descending order
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            
            # Create mask for top-p experts
            top_p_mask = cumsum <= self.top_p
            top_p_mask[..., 0] = True  # Always keep the top expert
            
            # Create allowed experts mask
            allowed = torch.zeros_like(probs, dtype=torch.bool)
            allowed.scatter_(1, sorted_idx, top_p_mask)
            
            # Mask out disallowed experts by setting their logits to -inf
            logits = logits.masked_fill(~allowed, float("-inf"))

        # ------------------------------------------------------------------
        # STANDARD ROUTING (now operates on potentially filtered logits)
        # ------------------------------------------------------------------
        if self.routing_type == "sinkhorn":
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                deterministic_mode=self.config.deterministic_mode,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map


class XMoERouter(TopKRouter):
    """X-MoE router (Chi et al., 2022, https://arxiv.org/abs/2204.09179).

    Extends TopKRouter with three modifications:
      1. Dimension reduction: project hidden states to a low-dim expert embedding space.
      2. L2 normalization of both projected tokens and expert embeddings (cosine routing).
      3. Learnable temperature scalar applied before softmax gating.

    Expert embeddings are initialized with L2 norm 0.1 and kept at that norm during
    training to prevent token-assignment fluctuations (see paper §3.1).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.hidden_size
        # de defaults to N/2 per the paper; override via xmoe_expert_dim config field
        expert_dim = getattr(config, 'xmoe_expert_dim', None) or (config.num_moe_experts // 2)

        # Replace the base Router's weight (which assumed full hidden_size) with:
        #   proj_weight:       W  ∈ R^{de × d}  (projects tokens into expert space)
        #   expert_embeddings: e_i ∈ R^{de}     (one per expert)
        del self.weight  # not used by this router

        self.proj_weight = nn.Parameter(
            torch.empty((expert_dim, hidden_size), dtype=torch.float32)
        )
        config.init_method(self.proj_weight)
        self.proj_weight.data = self.proj_weight.data.to(dtype=config.params_dtype)
        setattr(self.proj_weight, 'sequence_parallel', config.sequence_parallel)

        self.expert_embeddings = nn.Parameter(
            torch.empty((config.num_moe_experts, expert_dim), dtype=torch.float32)
        )
        nn.init.normal_(self.expert_embeddings)
        with torch.no_grad():
            self.expert_embeddings.data = F.normalize(self.expert_embeddings.data, dim=-1) * 0.1
        self.expert_embeddings.data = self.expert_embeddings.data.to(dtype=config.params_dtype)
        setattr(self.expert_embeddings, 'sequence_parallel', config.sequence_parallel)

        # Learnable temperature (paper Eq. 8); init to 1.0
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.temperature.data = self.temperature.data.to(dtype=config.params_dtype)

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """Cosine routing scores with dimension reduction and L2 normalization."""
        # Keep expert embeddings pinned at norm 0.1 throughout training
        with torch.no_grad():
            self.expert_embeddings.data = F.normalize(self.expert_embeddings.data, dim=-1) * 0.1

        projected = F.linear(input.view(-1, input.shape[-1]), self.proj_weight)  # [T, de]
        projected = F.normalize(projected, dim=-1)                                # unit sphere
        # expert_embeddings norm is pinned to 0.1; divide by constant to get unit vectors.
        # Using /0.1 instead of F.normalize keeps the gradient clean (no normalization Jacobian).
        embeddings = self.expert_embeddings / 0.1                                 # unit sphere
        self._raw_cosine_scores = projected @ embeddings.t()                      # [T, N], τ-free
        return self._raw_cosine_scores / self.temperature.abs().clamp(min=1e-6)

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Load balancing with fixed-temperature aux loss, per paper Eq. 9.

        Top-k routing uses τ-scaled logits (learnable temperature).
        The aux loss uses raw cosine scores (τ₀ = 1, fixed) so the balancing
        signal is decoupled from the learned temperature scale.
        """
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )

        if self.training:
            # Use raw cosine scores (τ₀=1) for the aux loss, not the τ-scaled logits
            aux_scores = torch.softmax(self._raw_cosine_scores.float(), dim=-1)
            probs = self.apply_load_balancing_loss(aux_scores, tokens_per_expert, activation=probs)
        return probs, routing_map


class ReLURouter(Router):
    """Route each token to the experts with non-zero relu outputs."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the relu router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        # self.target_sparsity = 1 - self.topk / self.num_experts
        self.input_jitter = None

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def l1_reg_load_balancing(self, logits: torch.Tensor):
        """Apply load balancing L1 regularization loss to the ReLU output.

        Args:
            logits (torch.Tensor): Logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment, shape [num_tokens, num_experts].
            routing_map (torch.Tensor): The mapping of token to experts assignment, shape [num_tokens, num_experts].
        """
        probs = torch.relu(logits)
        routing_map = probs > 0
        if self.training and torch.is_grad_enabled():
            num_local_tokens_per_expert = routing_map.sum(dim=0)
            # Apply l1 regularization
            probs = self.apply_l1_reg(probs, num_local_tokens_per_expert, activation=probs)
            # Record the sparsity of the ReLU output
            sparsity = 1 - routing_map.sum().float() / routing_map.numel()
            self.config.moe_relu_sparsity += sparsity
        return probs, routing_map

    def apply_l1_reg(self, probs: torch.Tensor, num_local_tokens_per_expert: torch.Tensor, activation: torch.Tensor):
        """Apply load balancing L1 regularization loss to the ReLU output.

        Args:
            probs (torch.Tensor): The probs output by the router for each token.
                [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert.
                [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        l1_reg_coeff = self.config.moe_relu_l1_reg_coeff.item()

        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            l1_reg_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        # L1 regularization with load balancing shares the same formula with switch load balancing loss:
        # l1_reg = sum((probs_per_expert/num_tokens) *
        # (tokens_per_expert/(num_tokens*topk))) * num_experts * l1_reg_coeff.
        l1_reg = switch_load_balancing_loss_func(
            probs,
            num_local_tokens_per_expert,
            self.topk,
            l1_reg_coeff,
            sequence_partition_group=sequence_partition_group,
        )

        save_to_aux_losses_tracker(
            "l1_reg_loss",
            l1_reg / l1_reg_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, l1_reg)
        return activation

    def routing(self, logits: torch.Tensor):
        """ReLU routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment, shape [num_tokens, num_experts].
            routing_map (torch.Tensor): The mapping of token to experts assignment, shape [num_tokens, num_experts].
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        scores, routing_map = self.l1_reg_load_balancing(logits)

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map



class CoRM(Router):
    """
    CoRM: Contrastive Routing Mixture-of-Experts Router.

    Each expert is an independent mechanism that attends to incoming tokens.
    Routing is determined by which expert's attention shifts most when seeing
    the real token vs an average token (null baseline).

    Architecture:
        K(x)      = normalize(key(LN(x)))           — shared key: "what this token is"
        Q_e(x)    = normalize(expert_queries(LN(x))) — per-expert query from real input
        Q_e(null) = normalize(expert_queries(mu))    — per-expert query from EMA null
                    (constant per expert, different across experts)

    Routing logit for expert e on token x:
        attn_real[e] = Q_e(x)    · K(x) / sqrt(d2)
        attn_null[e] = Q_e(null) · K(x) / sqrt(d2)
        gap[e]       = attn_real[e] - attn_null[e]
                     = [Q_e(x) - Q_e(null)] · K(x) / sqrt(d2)

    Interpretation: "How much does expert e's query shift toward this token's
    key, compared to expert e's resting-state query?" The expert whose query
    shifts the most wins the token.

    Null state: EMA of LayerNorm'd hidden states (detached, no gradient).
    Stored in H-space, projected through live expert_queries weights.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        if hasattr(self, "weight"):
            del self._parameters["weight"]

        # Standard attributes
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type

        # CoRM-specific configuration
        self.expert_attn_size = getattr(config, 'expert_attn_size', 64)
        self.use_dynamic_routing = getattr(config, 'use_dynamic_routing', False)
        self.top_p = getattr(config, 'top_p', 1.0)
        # Ablation: whether to L2-normalize K, Q_real, Q_null before attention.
        # Default True (paper's setup). Set False via --moe-router-no-l2-normalize.
        self.use_l2_normalize = getattr(config, 'moe_router_use_l2_normalize', True)

        # Dimensions
        d1 = self.config.hidden_size
        d2 = self.expert_attn_size
        E = self.config.num_moe_experts
        self.num_experts = E

        # === 1. Null State (EMA of LN'd hidden states) ===
        # Stored as a non-trainable Parameter (requires_grad=False) so that
        # Megatron's DCP checkpoint saver includes it automatically.
        self.global_null_mean = nn.Parameter(torch.zeros(d1), requires_grad=False)
        # Plain bool — not a Parameter/buffer. DCP would ALLGATHER bool scalars
        # across ranks and deadlock. The EMA value itself (global_null_mean) is saved;
        # this flag just tells us whether to initialise or update on the first batch.
        self._ema_initialized = False
        self.momentum = getattr(config, 'moe_ema_null_momentum',
                                getattr(config, 'router_background_momentum', 0.01))

        # === 2. Shared Key Projection: what the token IS ===
        self.key = torch.nn.Linear(d1, d2, bias=False)

        # === 3. Per-Expert Query Projection: how each expert reads the token ===
        self.expert_queries = torch.nn.Linear(d1, E * d2, bias=False)

        # Initialization
        if config.perform_initialization:
            config.init_method(self.key.weight)
            config.init_method(self.expert_queries.weight)

        self.routing_temperature = nn.Parameter(torch.ones(1))

        if config.sequence_parallel:
            setattr(self.key.weight, 'sequence_parallel', True)
            setattr(self.expert_queries.weight, 'sequence_parallel', True)

        # Health logging — use class-level step counter so all layers share it
        self._log_interval = getattr(config, 'moe_log_interval', 100)
        if not hasattr(CoRM, '_global_fwd_count'):
            CoRM._global_fwd_count = 0
            CoRM._num_moe_layers = 0
        CoRM._num_moe_layers += 1

    def _log_health(self, attn_real, attn_null, gap, Q_real, Q_null, K):
        """
        Log router health metrics. Only rank 0, every _log_interval training steps.

        Uses a global forward counter shared across all router instances so that
        we count actual training iterations (not per-layer forwards). All 12 layers
        accumulate stats, and the LAST layer to fire in each iteration prints them all.
        """
        if not self.training:
            return
        # Strict rank-0 only: must be initialized and rank must be exactly 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() != 0:
            return

        cls = CoRM

        # Count global forwards. Each training step calls gating() once per MoE layer.
        # Increment the counter; when it hits num_moe_layers, one full iteration is done.
        cls._global_fwd_count += 1
        iteration = cls._global_fwd_count // cls._num_moe_layers  # 0-indexed training step
        layer_idx_in_step = cls._global_fwd_count % cls._num_moe_layers  # which layer in this step

        # Try to use Megatron's authoritative iteration counter if available
        try:
            from megatron.training import get_args
            args = get_args()
            if hasattr(args, 'curr_iteration') and args.curr_iteration is not None:
                iteration = args.curr_iteration
        except Exception:
            pass

        # Always accumulate stats (cheap under no_grad)
        with torch.no_grad():
            layer = self.layer_number if self.layer_number is not None else layer_idx_in_step

            gap_mean = gap.mean(0).cpu()
            gap_std = gap.std(0).cpu()
            ar_mean = attn_real.mean(0).cpu()
            an_mean = attn_null.mean(0).cpu()
            discrim = gap.std(dim=-1).mean().item()
            cos_sim = (Q_real * Q_null.unsqueeze(0)).sum(-1).mean(0).cpu()
            frac_pos = (gap > 0).float().mean(0).cpu()
            temp_val = self.routing_temperature.item()

            if not hasattr(cls, '_current_step_stats'):
                cls._current_step_stats = {}
            cls._current_step_stats[layer] = {
                'gap_mean': gap_mean, 'gap_std': gap_std,
                'ar_mean': ar_mean, 'an_mean': an_mean,
                'discrim': discrim, 'cos_sim': cos_sim,
                'frac_pos': frac_pos, 'temp': temp_val,
            }

            # Only print when the last layer of this iteration fires
            if layer_idx_in_step != 0:
                # layer_idx_in_step == 0 means we just wrapped → previous step is complete
                return

            # Check if this is a logging step
            if iteration % self._log_interval != 0 or iteration == 0:
                cls._current_step_stats = {}
                return

            # Dedup: don't print the same iteration twice (guards against multiple processes
            # or multiple triggers reaching this point)
            if not hasattr(cls, '_last_printed_iter'):
                cls._last_printed_iter = -1
            if iteration == cls._last_printed_iter:
                return
            cls._last_printed_iter = iteration

            current = cls._current_step_stats
            if not current:
                return

            # --- Print summary ---
            layers_sorted = sorted(current.keys())
            import sys
            print(f"\n{'='*90}", flush=True)
            print(f"[Iter {iteration}] CoRM Router Health — Contrastive Attention Diagnostics", flush=True)
            print(f"{'='*90}", flush=True)

            print(f"{'L':<3} | {'Temp':<5} | {'Discrim':<7} | "
                  f"{'Gap mu (per expert)':<36} | {'FracPos (per expert)':<36}", flush=True)
            print(f"{'-'*3}-+-{'-'*5}-+-{'-'*7}-+-{'-'*36}-+-{'-'*36}", flush=True)

            for li in layers_sorted:
                s = current[li]
                gm = ",".join([f"{v:+.3f}" for v in s['gap_mean'].tolist()])
                fp = ",".join([f"{v:.2f}" for v in s['frac_pos'].tolist()])
                print(f"{li:<3} | {s['temp']:.2f}  | {s['discrim']:.4f}  | [{gm}] | [{fp}]", flush=True)

            # Detailed view for 3 representative layers
            focal = [layers_sorted[0], layers_sorted[len(layers_sorted)//2], layers_sorted[-1]]
            focal = sorted(set(focal))
            print(f"\n--- Detailed (layers {focal}) ---", flush=True)
            for li in focal:
                s = current[li]
                ar = ",".join([f"{v:.3f}" for v in s['ar_mean'].tolist()])
                an = ",".join([f"{v:.3f}" for v in s['an_mean'].tolist()])
                cs = ",".join([f"{v:.3f}" for v in s['cos_sim'].tolist()])
                gs = ",".join([f"{v:.3f}" for v in s['gap_std'].tolist()])
                print(f"\n  Layer {li}:", flush=True)
                print(f"    attn_real  mu: [{ar}]", flush=True)
                print(f"    attn_null  mu: [{an}]", flush=True)
                print(f"    gap std      : [{gs}]", flush=True)
                print(f"    Q cos(r,n)   : [{cs}]", flush=True)

            print(f"{'='*90}\n", flush=True)
            sys.stdout.flush()

            # Clean up for next logging window
            cls._current_step_stats = {}

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            batch_size, seq_len, hidden_size = input.shape
            input = input.view(-1, hidden_size)

        N, H = input.shape
        E = self.num_experts
        d2 = self.expert_attn_size

        # === 1. LayerNorm the input ===
        real_n = F.layer_norm(input, (H,), eps=1e-5)

        # === 2. Update Null State EMA (training only, detached) ===
        # Store EMA of post-LN representations so null lives in the same space as real_n
        if self.training:
            with torch.no_grad():
                batch_mean = real_n.mean(dim=0).detach()
                if not self._ema_initialized:
                    self.global_null_mean.copy_(batch_mean)
                    self._ema_initialized = True
                else:
                    self.global_null_mean.mul_(1 - self.momentum).add_(
                        self.momentum * batch_mean
                    )

        # === 3. Shared Key: what this token is ===
        K = self.key(real_n)                                    # [N, d2]
        if self.use_l2_normalize:
            K = F.normalize(K, p=2, dim=-1)

        # === 4. Per-Expert Queries ===
        # Real queries: each expert reads THIS token
        Q_real = self.expert_queries(real_n).view(N, E, d2)     # [N, E, d2]
        if self.use_l2_normalize:
            Q_real = F.normalize(Q_real, p=2, dim=-1)

        # Null queries: each expert reads the AVERAGE token (constant per expert)
        # Projected through live weights — stays current as model trains
        Q_null = self.expert_queries(self.global_null_mean).view(E, d2)  # [E, d2]
        if self.use_l2_normalize:
            Q_null = F.normalize(Q_null, p=2, dim=-1)

        # === 5. Contrastive Attention ===
        scale = 1.0 / math.sqrt(d2)

        # attn_real[n,e] = Q_e(x_n) · K(x_n) — "expert e attends to this token"
        attn_real = (Q_real * K.unsqueeze(1)).sum(-1) * scale   # [N, E]

        # attn_null[n,e] = Q_e(null) · K(x_n) — "expert e's baseline attention to this token"
        attn_null = (Q_null.unsqueeze(0) * K.unsqueeze(1)).sum(-1) * scale  # [N, E]

        # gap = how much MORE expert e attends to real than its baseline
        gap = attn_real - attn_null                              # [N, E]

        # Equivalently: [Q_e(x) - Q_e(null)] · K(x) / sqrt(d2)
        # = expert e's "query shift" projected onto this token's key

        # === 6. Temperature and Stability ===
        temp = torch.clamp(self.routing_temperature, min=0.1, max=10.0)
        router_logits = gap * temp
        router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)

        # === 7. Health logging ===
        self._log_health(attn_real, attn_null, gap, Q_real, Q_null, K)

        return router_logits

    def apply_input_jitter(self, input: torch.Tensor) -> torch.Tensor:
        """Add noise to the input tensor. Refer to https://arxiv.org/abs/2101.03961."""
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if getattr(self, "input_jitter", None) is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def apply_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Encourages the router's logits to remain small to enhance stability."""
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            # Assuming z_loss_func, MoEAuxLossAutoScaler, save_to_aux_losses_tracker are imported
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """Applies auxiliary loss to the MoE layer."""
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        # Assuming switch_load_balancing_loss_func is imported
        aux_loss = switch_load_balancing_loss_func(
            probs,
            num_local_tokens_per_expert,
            self.topk,
            moe_aux_loss_coeff,
            sequence_partition_group=sequence_partition_group,
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CoRM routing function with optional auxiliary load balancing loss."""
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Assuming gather_from_sequence_parallel_region is imported
            logits = gather_from_sequence_parallel_region(logits)

        # ------------------------------------------------------------------
        # OPTIONAL TOP-P (eligibility mask, NOT routing)
        # ------------------------------------------------------------------
        if self.use_dynamic_routing and self.top_p < 1.0:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            top_p_mask = cumsum <= self.top_p
            top_p_mask[..., 0] = True  # always keep top expert

            allowed = torch.zeros_like(probs, dtype=torch.bool)
            allowed.scatter_(1, sorted_idx, top_p_mask)

            # Mask logits so disallowed experts can never be selected
            logits = logits.masked_fill(~allowed, float("-inf"))

        # ------------------------------------------------------------------
        # CAPACITY-AWARE ROUTING
        # ------------------------------------------------------------------
        use_pre_softmax = (self.topk == 1)

        # Assuming topk_softmax_with_capacity is imported
        scores, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=use_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )

        # ------------------------------------------------------------------
        # AUXILIARY LOAD BALANCING LOSS (TRAINING ONLY)
        # ------------------------------------------------------------------
        if self.routing_type == "aux_loss" and self.training:
            scores_for_loss = torch.softmax(logits, dim=-1, dtype=torch.float32)
            scores = self.apply_load_balancing_loss(
                scores_for_loss,
                tokens_per_expert,
                activation=scores,
            )

        return scores, routing_map
        
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.hidden = input.shape[-1]

        input = self.apply_input_jitter(input)

        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map
        
