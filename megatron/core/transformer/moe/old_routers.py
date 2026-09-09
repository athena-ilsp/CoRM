"""
Archive of deprecated router variants. Not used in active training.

These router classes were previously in router.py but are no longer used.
They are kept here for reference only.

Deprecated routers:
- Shared_RIMRouter_shared_null: RIM Router with shared expert as null state
- SharedRIMRouter_DualNull_NoV_Filtered: Dual null + filtered input reconstruction (v1, log-prob)
- SharedRIMRouter_DualNull_NoV_Filtered_2: Dual null + filtered input reconstruction (v2, raw cosine)
- SharedRIMRouter_v3_ema_no_V: EMA null state + Q,K-only (no V) + optional filtered inputs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


# ---------------------------------
---------------------------------
# RIM Router with shared expert creating null
class Shared_RIMRouter_shared_null(Router):
    """
    RIM Router with same-layer shared expert as null state.
    
    Routing score: (a_real - a_null) + (a_real - a_null_shared)
    
    Where:
    - a_null: learnable baseline
    - a_null_shared: current layer's shared expert output (computed first)
    
    Routed experts must find patterns that:
    1. Are meaningful (beat null baseline)
    2. Are NOT captured by the shared expert (beat shared output)
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
        
        if hasattr(self, "weight"):
            del self._parameters["weight"]

        # Standard router attributes
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None
        
        # RIM-specific configuration
        self.expert_attn_size = getattr(config, 'expert_attn_size', 64)
        self.use_dynamic_routing = getattr(config, 'use_dynamic_routing', False)
        self.top_p = getattr(config, 'top_p', 1.0)
        
        # Dimensions
        d1 = self.config.hidden_size
        d2 = self.expert_attn_size
        E = self.num_experts
        ED = E * d2

        # === Null State (learnable baseline) ===
        self.register_buffer('null_state', torch.zeros(d1))
        
        # === Shared K, V Projections ===
        self.key = nn.Linear(d1, d2, bias=True)
        self.value = nn.Linear(d1, d2, bias=True)
        
        # === Per-Expert Latent and Query Projections ===
        self.expert_states_flat = nn.Linear(d1, ED, bias=True)
        self.expert_query = nn.Linear(ED, ED, bias=True)
        
        # Initialize projections
        if config.perform_initialization:
            config.init_method(self.expert_states_flat.weight)
            config.init_method(self.key.weight)
            config.init_method(self.value.weight)
            config.init_method(self.expert_query.weight)
            
        with torch.no_grad():
            self.expert_query.bias.zero_()
        
        # Set dtype
        self.expert_states_flat.weight.data = self.expert_states_flat.weight.data.to(dtype=config.params_dtype)
        self.key.weight.data = self.key.weight.data.to(dtype=config.params_dtype)
        self.value.weight.data = self.value.weight.data.to(dtype=config.params_dtype)
        self.expert_query.weight.data = self.expert_query.weight.data.to(dtype=config.params_dtype)
        
        self.routing_temperature = nn.Parameter(torch.ones(1))

        # Sequence parallel
        if config.sequence_parallel:
            setattr(self.expert_states_flat.weight, 'sequence_parallel', True)
            setattr(self.key.weight, 'sequence_parallel', True)
            setattr(self.value.weight, 'sequence_parallel', True)
            setattr(self.expert_query.weight, 'sequence_parallel', True)

        self.use_learnable_weights = False
        if getattr(config, 'learn_shared_null_weight')==True:
            self.shared_null_weight = nn.Parameter(torch.zeros(1))  # Starts at 0.5/0.5
            self.use_learnable_weights = True
            print(f">>> DEBUG: [Using learnable shared_null_weight] ")

        elif getattr(config, 'fixed_shared_null_weight')== True:
            self.shared_null_weight = getattr(config, 'shared_null_weight', 0.5)
            print(f">>> DEBUG: [Using fixed shared_null_weight = {self.shared_null_weight}] ")

        else:
            self.shared_null_weight = 0.5
            print(f">>> DEBUG: [Using default (0.5) shared_null_weight] ")


    def _compute_activation(self, input_n: torch.Tensor) -> torch.Tensor:
        """
        Compute expert activations for normalized input.
        
        Args:
            input_n: Layer-normed input [N, H]
            
        Returns:
            activations: [N, E] activation per expert
        """
        N = input_n.shape[0]
        E = self.num_experts
        d2 = self.expert_attn_size
        
        # Per-expert latent states and queries
        H = self.expert_states_flat(input_n).view(N, E, d2)
        Q = self.expert_query(H.reshape(N, E * d2)).view(N, E, d2)
        
        # Shared K, V
        K = self.key(input_n)
        V = self.value(input_n)
        
        # Normalize Q, K
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        
        # Attention scores
        scale = 1.0 / math.sqrt(d2)
        scores = (Q * K.unsqueeze(1)).sum(-1) * scale  # [N, E]
        scores = scores - scores.max(-1, keepdim=True)[0]
        probs = F.softmax(scores, dim=-1)  # [N, E]
        
        # Activation magnitude
        a = (probs.unsqueeze(-1) * V.unsqueeze(1)).norm(p=2, dim=-1)  # [N, E]
        return a

    def gating(self, input: torch.Tensor, null_shared: torch.Tensor = None) -> torch.Tensor:
        """Memory-efficient RIM gating - only compute what's necessary."""

        # Debug logging (once per instance)
        # if not hasattr(self, '_debug_logged'):
        #     if null_shared is not None:
        #         print(f">>> DEBUG: [Layer {getattr(self, 'layer_number', '?')}] "
        #             f"Shared Router ACTIVE. null_shared shape: {null_shared.shape}")
        #     else:
        #         print(f">>> WARNING: [Layer {getattr(self, 'layer_number', '?')}] "
        #             f"Shared Router received None for null_shared!")
        #     self._debug_logged = True

        if input.dim() == 3:
            batch_size, seq_len, hidden_size = input.shape
            input = input.view(-1, hidden_size)
            if null_shared is not None:
                null_shared = null_shared.view(-1, hidden_size)
        
        N, H = input.shape
        orig_dtype = input.dtype
        
        # === Only compute a_real with gradients ===
        real_n = F.layer_norm(input, (H,), eps=1e-5)
        a_real = self._compute_activation(real_n)
        
        # === a_null: zeros baseline (no gradient needed) ===
        # null = self.null_state.to(orig_dtype).expand(N, H)
        null = self.null_state.to(orig_dtype).expand(N, H)
        null_n = F.layer_norm(null, (H,), eps=1e-5)
        with torch.no_grad():
            a_null = self._compute_activation(null_n)
        
        # === Start with base score ===
        router_logits = a_real - a_null
        
        # === Add shared expert term (no gradient needed) ===
        if null_shared is not None:
            null_shared_n = F.layer_norm(null_shared, (H,), eps=1e-5)
            with torch.no_grad():
                a_null_shared = self._compute_activation(null_shared_n)
            
            if self.use_learnable_weights:
                shared_weight = torch.sigmoid(self.shared_null_weight)
            else:
                shared_weight = self.shared_null_weight

            null_weight = 1 - shared_weight
            router_logits = null_weight * router_logits + shared_weight * (a_real - a_null_shared)
        
        # === Finalize ===
        temp_clamped = torch.clamp(self.routing_temperature, min=0.1, max=10.0)
        router_logits = router_logits * temp_clamped
        router_logits = router_logits - router_logits.mean(-1, keepdim=True)
        router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)
        
        return router_logits
    
    def apply_input_jitter(self, input: torch.Tensor) -> torch.Tensor:
        """Add noise to input tensor."""
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        return input

    def apply_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply z-loss for stability."""
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

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """Apply auxiliary load balancing loss."""
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

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing from logits."""
        logits = logits.view(-1, self.config.num_moe_experts)

        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            logits = gather_from_sequence_parallel_region(logits)

        if self.use_dynamic_routing and self.top_p < 1.0:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            top_p_mask = cumsum <= self.top_p
            top_p_mask[..., 0] = True
            allowed = torch.zeros_like(probs, dtype=torch.bool)
            allowed.scatter_(1, sorted_idx, top_p_mask)
            logits = logits.masked_fill(~allowed, float("-inf"))

        use_pre_softmax = (self.topk == 1)

        scores, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=use_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )

        if self.routing_type == "aux_loss" and self.training:
            scores_for_loss = torch.softmax(logits, dim=-1, dtype=torch.float32)
            scores = self.apply_load_balancing_loss(
                scores_for_loss,
                tokens_per_expert,
                activation=scores,
            )

        return scores, routing_map

    def forward(
        self, input: torch.Tensor, null_shared: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input: Input tensor
            null_shared: Shared expert output from same layer
            
        Returns:
            scores: Routing probabilities
            routing_map: Expert assignments
        """
        self.hidden = input.shape[-1]

        input = self.apply_input_jitter(input)

        logits = self.gating(input, null_shared=null_shared)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map


---------------------------------
  Scheduled Router + Inversion
---------------------------------

Old version - had log p -> changed it to raw scores
class SharedRIMRouter_DualNull_NoV_Filtered(Router):
    """
    RIM Router with filtered input reconstruction and Residual Connection.
    
    - Projects tokens to latent space (64-dim)
    - Computes routing via Log-Likelihood Ratio (Real vs Null)
    - Projects back to original space
    - Returns (0.5 * Filtered + 0.5 * Original) for expert processing
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
        
        if hasattr(self, "weight"):
            del self._parameters["weight"]

        # Standard router attributes
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None
        
        # RIM-specific configuration
        self.expert_attn_size = getattr(config, 'expert_attn_size', 64)
        self.use_dynamic_routing = getattr(config, 'use_dynamic_routing', False)
        self.top_p = getattr(config, 'top_p', 1.0)
        
        # Filtered inputs configuration
        self.use_filtered_inputs = getattr(config, 'rim_use_filtered_inputs', True)
        self.use_ortho_reg = getattr(config, 'rim_ortho_reg', True)
        self.ortho_reg_coeff = getattr(config, 'rim_ortho_reg_coeff', 0.01)
        
        # Dimensions
        d1 = self.config.hidden_size
        d2 = self.expert_attn_size
        E = self.num_experts
        ED = E * d2

        # === Null State (learnable baseline) ===
        self.register_buffer('null_state', torch.zeros(d1))
        
        # === Shared K Projection ===
        self.key = nn.Linear(d1, d2, bias=True)
        
        # === Per-Expert Encoder (H -> d) ===
        self.expert_encoder = nn.Linear(d1, ED, bias=True)
        
        # === Per-Expert Query ===
        self.expert_query = nn.Linear(ED, ED, bias=True)
        
        # Initialize projections
        if config.perform_initialization:
            config.init_method(self.expert_encoder.weight)
            config.init_method(self.key.weight)
            config.init_method(self.expert_query.weight)
            
        with torch.no_grad():
            self.expert_query.bias.zero_()
        
        # Set dtype
        self.expert_encoder.weight.data = self.expert_encoder.weight.data.to(dtype=config.params_dtype)
        self.key.weight.data = self.key.weight.data.to(dtype=config.params_dtype)
        self.expert_query.weight.data = self.expert_query.weight.data.to(dtype=config.params_dtype)
        
        # Temperature parameters
        self.attention_temperature = nn.Parameter(torch.ones(1, dtype=config.params_dtype))
        self.routing_temperature = nn.Parameter(torch.ones(1, dtype=config.params_dtype))

        # Sequence parallel
        if config.sequence_parallel:
            setattr(self.expert_encoder.weight, 'sequence_parallel', True)
            setattr(self.key.weight, 'sequence_parallel', True)
            setattr(self.expert_query.weight, 'sequence_parallel', True)

        # === Weight buffers (updated by MoELayer scheduler) ===
        self.register_buffer('w_null', torch.tensor(0.5))
        self.register_buffer('w_shared', torch.tensor(0.5))

    def _compute_log_probs_and_latents(self, input_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Log-Probabilities (for routing) and Latents (for filtering).
        
        Args:
            input_n: Layer-normed input [N, H]
            
        Returns:
            log_probs: [N, E] log-probability per expert
            H_e: [N, E, d2] latent representations per expert
        """
        N = input_n.shape[0]
        E = self.num_experts
        d2 = self.expert_attn_size
        
        # Encode to latent space
        H_e = self.expert_encoder(input_n).view(N, E, d2)  # [N, E, 64]
        
        # Compute queries
        Q = self.expert_query(H_e.reshape(N, E * d2)).view(N, E, d2)
        
        # Shared key
        K = self.key(input_n)  # [N, 64]
        
        # Normalize
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        
        # Attention scores
        scale = 1.0 / math.sqrt(d2)
        scores = (Q * K.unsqueeze(1)).sum(-1) * scale  # [N, E]
        
        # === USE LOG-SOFTMAX ===
        # Use log_softmax to enable Likelihood Ratio comparison (Real - Null)
        temp = torch.clamp(self.attention_temperature, min=0.01, max=1.0)
        log_probs = F.log_softmax(scores / temp, dim=-1)
        
        return log_probs, H_e

    def _compute_filtered_inputs(self, H_e: torch.Tensor) -> torch.Tensor:
        """
        Project latent representations back to original dimension using transpose.
        """
        N, E, d2 = H_e.shape
        d1 = self.config.hidden_size
        
        # Get encoder weight: [E*d2, d1] -> [E, d2, d1]
        W = self.expert_encoder.weight.view(E, d2, d1)  # [E, 64, 768]
        
        # Project back using transpose: x_tilde = W^T * H_e
        filtered_inputs = torch.einsum('edh,ned->neh', W, H_e)  # [N, E, 768]
        
        return filtered_inputs

    def compute_orthonormality_loss(self) -> torch.Tensor:
        """Compute orthonormality regularization loss."""
        E = self.num_experts
        d2 = self.expert_attn_size
        d1 = self.config.hidden_size
        
        W = self.expert_encoder.weight.view(E, d2, d1)  # [E, 64, 768]
        
        # W * W^T should be identity [E, 64, 64]
        WWT = torch.bmm(W, W.transpose(-1, -2))
        
        # Target: identity matrix
        I = torch.eye(d2, device=W.device, dtype=W.dtype).unsqueeze(0).expand(E, -1, -1)
        
        # Frobenius norm
        ortho_loss = ((WWT - I) ** 2).mean()
        
        return ortho_loss

    def gating(self, input: torch.Tensor, null_shared: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing logits and filtered inputs.
        """
        # Handle 3D input
        if input.dim() == 3:
            input = input.view(-1, input.shape[-1])
            if null_shared is not None:
                null_shared = null_shared.view(-1, input.shape[-1])
        
        N, H = input.shape
        orig_dtype = input.dtype
        
        # === 1. Real Input: Log-Probs + Latents ===
        real_n = F.layer_norm(input, (H,), eps=1e-5)
        log_p_real, H_e_real = self._compute_log_probs_and_latents(real_n)
        
        # === 2. Compute Filtered Inputs + Residual ===
        filtered_inputs = None
        if self.use_filtered_inputs:
            # Reconstruct from latent
            reconstruction = self._compute_filtered_inputs(H_e_real)  # [N, E, H]
            
            # RESIDUAL STREAM: 50% Filtered + 50% Original
            # input is [N, H], need [N, 1, H] to broadcast to experts
            filtered_inputs = 0.5 * reconstruction + 0.5 * input.unsqueeze(1)
        
        # === 3. Null Log-Probs ===
        null_vec = self.null_state.to(orig_dtype).expand(N, H)
        null_n = F.layer_norm(null_vec, (H,), eps=1e-5)
        with torch.no_grad():
            log_p_null, _ = self._compute_log_probs_and_latents(null_n)
        
        # === 4. Compute Routing Logits (Likelihood Ratio) ===
        router_logits = self.w_null * (log_p_real - log_p_null)
        
        # Add shared expert term if available
        if null_shared is not None:
            shared_n = F.layer_norm(null_shared, (H,), eps=1e-5)
            with torch.no_grad():
                log_p_shared, _ = self._compute_log_probs_and_latents(shared_n)
            router_logits = router_logits + self.w_shared * (log_p_real - log_p_shared)
        
        # === 5. Finalize ===
        temp_clamped = torch.clamp(self.routing_temperature, min=0.1, max=10.0)
        router_logits = router_logits * temp_clamped
        router_logits = router_logits - router_logits.mean(-1, keepdim=True)
        router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)
        
        return router_logits, filtered_inputs

    # ... (Keep apply_input_jitter, apply_z_loss, apply_load_balancing_loss, apply_orthonormality_loss as is) ...
    def apply_input_jitter(self, input: torch.Tensor) -> torch.Tensor:
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        return input

    def apply_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
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

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
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

    def apply_orthonormality_loss(self, activation: torch.Tensor) -> torch.Tensor:
        """Apply orthonormality regularization loss."""
        if self.use_ortho_reg and self.training:
            ortho_loss = self.compute_orthonormality_loss()
            save_to_aux_losses_tracker(
                "orthonormality_loss",
                ortho_loss,
                self.layer_number,
                self.config.num_layers,
            )
            activation = MoEAuxLossAutoScaler.apply(activation, ortho_loss * self.ortho_reg_coeff)
        return activation

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = logits.view(-1, self.config.num_moe_experts)
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            logits = gather_from_sequence_parallel_region(logits)

        if self.use_dynamic_routing and self.top_p < 1.0:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            top_p_mask = cumsum <= self.top_p
            top_p_mask[..., 0] = True
            allowed = torch.zeros_like(probs, dtype=torch.bool)
            allowed.scatter_(1, sorted_idx, top_p_mask)
            logits = logits.masked_fill(~allowed, float("-inf"))

        use_pre_softmax = (self.topk == 1)

        scores, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=use_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )

        if self.routing_type == "aux_loss" and self.training:
            scores_for_loss = torch.softmax(logits, dim=-1, dtype=torch.float32)
            scores = self.apply_load_balancing_loss(
                scores_for_loss,
                tokens_per_expert,
                activation=scores,
            )
        
        # Apply orthonormality loss
        scores = self.apply_orthonormality_loss(scores)

        return scores, routing_map

    def forward(
        self, input: torch.Tensor, null_shared: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden = input.shape[-1]
        input = self.apply_input_jitter(input)

        logits, filtered_inputs = self.gating(input, null_shared=null_shared)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map, filtered_inputs

New version - uses raw scores + random dummy instead of zeros
class SharedRIMRouter_DualNull_NoV_Filtered_2(Router):
    """
    RIM Router with filtered input reconstruction and Residual Connection.
    
    - Projects tokens to latent space (64-dim)
    - Computes routing via Log-Likelihood Ratio (Real vs Null)
    - Projects back to original space
    - Returns (0.5 * Filtered + 0.5 * Original) for expert processing
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
        
        if hasattr(self, "weight"):
            del self._parameters["weight"]

        # Standard router attributes
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None
        
        # RIM-specific configuration
        self.expert_attn_size = getattr(config, 'expert_attn_size', 64)
        self.use_dynamic_routing = getattr(config, 'use_dynamic_routing', False)
        self.top_p = getattr(config, 'top_p', 1.0)
        
        # Filtered inputs configuration
        self.use_filtered_inputs = getattr(config, 'rim_use_filtered_inputs', True)
        self.use_ortho_reg = getattr(config, 'rim_ortho_reg', True)
        self.ortho_reg_coeff = getattr(config, 'rim_ortho_reg_coeff', 0.01)
        
        # Dimensions
        d1 = self.config.hidden_size
        d2 = self.expert_attn_size
        E = self.num_experts
        ED = E * d2

        gen = torch.Generator()
        gen.manual_seed(42) 

        # 2. Generate the Dummy Vector
        # We use a standard normal distribution. 
        # Because you LayerNorm it later, the exact variance (0.02 vs 1.0) 
        # doesn't technically matter, but 0.02 is standard for transformers.
        dummy_vec = torch.randn(d1, generator=gen) * 0.02

        # === Null State (learnable baseline) ===
        self.register_buffer('null_state', dummy_vec)
        
        # === Shared K Projection ===
        self.key = nn.Linear(d1, d2, bias=True)
        
        # === Per-Expert Encoder (H -> d) ===
        self.expert_encoder = nn.Linear(d1, ED, bias=True)
        
        # === Per-Expert Query ===
        self.expert_query = nn.Linear(ED, ED, bias=True)
        
        # Initialize projections
        if config.perform_initialization:
            config.init_method(self.expert_encoder.weight)
            config.init_method(self.key.weight)
            config.init_method(self.expert_query.weight)
            
        with torch.no_grad():
            self.expert_query.bias.zero_()
        
        # Set dtype
        self.expert_encoder.weight.data = self.expert_encoder.weight.data.to(dtype=config.params_dtype)
        self.key.weight.data = self.key.weight.data.to(dtype=config.params_dtype)
        self.expert_query.weight.data = self.expert_query.weight.data.to(dtype=config.params_dtype)
        
        # Temperature parameters
        self.attention_temperature = nn.Parameter(torch.ones(1, dtype=config.params_dtype))
        self.routing_temperature = nn.Parameter(torch.ones(1, dtype=config.params_dtype))

        # Sequence parallel
        if config.sequence_parallel:
            setattr(self.expert_encoder.weight, 'sequence_parallel', True)
            setattr(self.key.weight, 'sequence_parallel', True)
            setattr(self.expert_query.weight, 'sequence_parallel', True)

        # === Weight buffers (updated by MoELayer scheduler) ===
        self.register_buffer('w_null', torch.tensor(0.5))
        self.register_buffer('w_shared', torch.tensor(0.5))

    def _compute_scores(self, input_n: torch.Tensor) -> torch.Tensor:
        """
        Compute raw cosine similarity scores using only Q and K (no V, no softmax).

        Args:
            input_n: Layer-normed input [N, H]

        Returns:
            scores: [N, E] cosine similarity per expert, bounded in [-1, 1]
        """
        N = input_n.shape[0]
        E = self.num_experts
        d2 = self.expert_attn_size

        # Encode to latent space
        H_e = self.expert_encoder(input_n).view(N, E, d2)  # [N, E, d2]

        # Compute per-expert queries
        Q = self.expert_query(H_e.reshape(N, E * d2)).view(N, E, d2)

        # Shared key
        K = self.key(input_n)  # [N, d2]

        # L2 normalize -> cosine similarity
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Raw cosine similarity (no scale, no softmax)
        # Q·K for unit vectors gives cosine similarity in [-1, 1]
        scores = (Q * K.unsqueeze(1)).sum(-1)  # [N, E]

        return scores

    def _compute_scores_and_latents(self, input_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute raw cosine similarity scores and latents (for filtering).

        Args:
            input_n: Layer-normed input [N, H]

        Returns:
            scores: [N, E] cosine similarity per expert, bounded in [-1, 1]
            H_e: [N, E, d2] latent representations per expert
        """
        N = input_n.shape[0]
        E = self.num_experts
        d2 = self.expert_attn_size

        # Encode to latent space
        H_e = self.expert_encoder(input_n).view(N, E, d2)  # [N, E, d2]

        # Compute per-expert queries
        Q = self.expert_query(H_e.reshape(N, E * d2)).view(N, E, d2)

        # Shared key
        K = self.key(input_n)  # [N, d2]

        # L2 normalize -> cosine similarity
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Raw cosine similarity (no scale, no softmax)
        scores = (Q * K.unsqueeze(1)).sum(-1)  # [N, E]

        return scores, H_e
    
    def _compute_filtered_inputs(self, H_e: torch.Tensor) -> torch.Tensor:
        """
        Project latent representations back to original dimension using transpose.
        """
        N, E, d2 = H_e.shape
        d1 = self.config.hidden_size
        
        # Get encoder weight: [E*d2, d1] -> [E, d2, d1]
        W = self.expert_encoder.weight.view(E, d2, d1)  # [E, 64, 768]
        
        # Project back using transpose: x_tilde = W^T * H_e
        filtered_inputs = torch.einsum('edh,ned->neh', W, H_e)  # [N, E, 768]
        
        return filtered_inputs

    def compute_orthonormality_loss(self) -> torch.Tensor:
        """Compute orthonormality regularization loss."""
        E = self.num_experts
        d2 = self.expert_attn_size
        d1 = self.config.hidden_size
        
        W = self.expert_encoder.weight.view(E, d2, d1)  # [E, 64, 768]
        
        # W * W^T should be identity [E, 64, 64]
        WWT = torch.bmm(W, W.transpose(-1, -2))
        
        # Target: identity matrix
        I = torch.eye(d2, device=W.device, dtype=W.dtype).unsqueeze(0).expand(E, -1, -1)
        
        # Frobenius norm
        ortho_loss = ((WWT - I) ** 2).mean()
        
        return ortho_loss

    def gating(self, input: torch.Tensor, null_shared: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing logits and filtered inputs.
        """
        # Handle 3D input
        if input.dim() == 3:
            input = input.view(-1, input.shape[-1])
            if null_shared is not None:
                null_shared = null_shared.view(-1, input.shape[-1])
        
        N, H = input.shape
        orig_dtype = input.dtype
        
        # === 1. Real Input: Log-Probs + Latents ===
        real_n = F.layer_norm(input, (H,), eps=1e-5)
        scores_real, H_e_real = self._compute_scores_and_latents(real_n)
        
        # === 2. Compute Filtered Inputs + Residual ===
        filtered_inputs = None
        if self.use_filtered_inputs:
            # Reconstruct from latent
            reconstruction = self._compute_filtered_inputs(H_e_real)  # [N, E, H]
            
            # RESIDUAL STREAM: 50% Filtered + 50% Original
            # input is [N, H], need [N, 1, H] to broadcast to experts
            filtered_inputs = 0.5 * reconstruction + 0.5 * input.unsqueeze(1)
        
        # === 3. Null Log-Probs ===
        null_vec = self.null_state.to(orig_dtype).expand(N, H)
        null_n = F.layer_norm(null_vec, (H,), eps=1e-5)
        with torch.no_grad():
            scores_null, _ = self._compute_scores_and_latents(null_n)
        
        # === 4. Compute Routing Logits (Likelihood Ratio) ===
        router_logits = self.w_null * (scores_real - scores_null)
        
        # Add shared expert term if available
        if null_shared is not None:
            shared_n = F.layer_norm(null_shared, (H,), eps=1e-5)
            with torch.no_grad():
                scores_shared, _ = self._compute_scores_and_latents(shared_n)
            router_logits = router_logits + self.w_shared * (scores_real - scores_shared)
        
        # === 5. Finalize ===
        temp_clamped = torch.clamp(self.routing_temperature, min=0.1, max=10.0)
        router_logits = router_logits * temp_clamped
        router_logits = router_logits - router_logits.mean(-1, keepdim=True)
        router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)
        
        return router_logits, filtered_inputs

    # ... (Keep apply_input_jitter, apply_z_loss, apply_load_balancing_loss, apply_orthonormality_loss as is) ...
    def apply_input_jitter(self, input: torch.Tensor) -> torch.Tensor:
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        return input

    def apply_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
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

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
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

    def apply_orthonormality_loss(self, activation: torch.Tensor) -> torch.Tensor:
        """Apply orthonormality regularization loss."""
        if self.use_ortho_reg and self.training:
            ortho_loss = self.compute_orthonormality_loss()
            save_to_aux_losses_tracker(
                "orthonormality_loss",
                ortho_loss,
                self.layer_number,
                self.config.num_layers,
            )
            activation = MoEAuxLossAutoScaler.apply(activation, ortho_loss * self.ortho_reg_coeff)
        return activation

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = logits.view(-1, self.config.num_moe_experts)
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            logits = gather_from_sequence_parallel_region(logits)

        if self.use_dynamic_routing and self.top_p < 1.0:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            top_p_mask = cumsum <= self.top_p
            top_p_mask[..., 0] = True
            allowed = torch.zeros_like(probs, dtype=torch.bool)
            allowed.scatter_(1, sorted_idx, top_p_mask)
            logits = logits.masked_fill(~allowed, float("-inf"))

        use_pre_softmax = (self.topk == 1)

        scores, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=use_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )

        if self.routing_type == "aux_loss" and self.training:
            scores_for_loss = torch.softmax(logits, dim=-1, dtype=torch.float32)
            scores = self.apply_load_balancing_loss(
                scores_for_loss,
                tokens_per_expert,
                activation=scores,
            )
        
        # Apply orthonormality loss
        scores = self.apply_orthonormality_loss(scores)

        return scores, routing_map

    def forward(
        self, input: torch.Tensor, null_shared: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden = input.shape[-1]
        input = self.apply_input_jitter(input)

        logits, filtered_inputs = self.gating(input, null_shared=null_shared)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map, filtered_inputs


---------------------------------
v3 EMA Router WITHOUT V Matrix (Q,K only)
---------------------------------

class SharedRIMRouter_v3_ema_no_V(Router):
    """
    RIM-style router combining EMA null state with Q,K-only attention (no V matrix).

    Key Features:
    - Uses EMA (Exponential Moving Average) as null_state for BOTH training and inference
    - Only uses Q, K matrices for attention scores (no V matrix)
    - Routing via raw cosine similarity difference: score(real) - score(ema_null)
    - Scores are bounded in [-1, 1] (cosine similarity), difference in [-2, 2]
    - Optional filtered inputs via encoder transpose projection
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)

        if hasattr(self, "weight"):
            del self._parameters["weight"]

        # Standard router attributes
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None

        # RIM-specific configuration
        self.expert_attn_size = getattr(config, 'expert_attn_size', 64)
        self.use_dynamic_routing = getattr(config, 'use_dynamic_routing', False)
        self.top_p = getattr(config, 'top_p', 1.0)

        # Filtered inputs configuration
        self.use_filtered_inputs = getattr(config, 'rim_use_filtered_inputs', False)
        self.use_ortho_reg = getattr(config, 'rim_ortho_reg', True)
        self.ortho_reg_coeff = getattr(config, 'rim_ortho_reg_coeff', 0.01)

        # EMA configuration
        self.momentum = getattr(config, 'router_background_momentum', 0.01)

        # Dimensions
        d1 = self.config.hidden_size
        d2 = self.expert_attn_size
        E = self.num_experts
        ED = E * d2

        # === EMA Null State (Layer-Specific Background) ===
        self.global_null_mean = nn.Parameter(
            torch.zeros(1, d1),
            requires_grad=False  # Updated via EMA, not gradients
        )

        # === Shared K Projection (NO V!) ===
        self.key = nn.Linear(d1, d2, bias=True)

        # === Per-Expert Encoder (H -> d) ===
        self.expert_encoder = nn.Linear(d1, ED, bias=True)

        # === Per-Expert Query ===
        self.expert_query = nn.Linear(ED, ED, bias=True)

        # Initialize projections
        if config.perform_initialization:
            config.init_method(self.expert_encoder.weight)
            config.init_method(self.key.weight)
            config.init_method(self.expert_query.weight)

        with torch.no_grad():
            self.expert_query.bias.zero_()

        # Set dtype
        self.expert_encoder.weight.data = self.expert_encoder.weight.data.to(dtype=config.params_dtype)
        self.key.weight.data = self.key.weight.data.to(dtype=config.params_dtype)
        self.expert_query.weight.data = self.expert_query.weight.data.to(dtype=config.params_dtype)

        # Temperature parameters
        self.attention_temperature = nn.Parameter(torch.ones(1, dtype=config.params_dtype))
        self.routing_temperature = nn.Parameter(torch.ones(1, dtype=config.params_dtype))

        # Sequence parallel
        if config.sequence_parallel:
            setattr(self.expert_encoder.weight, 'sequence_parallel', True)
            setattr(self.key.weight, 'sequence_parallel', True)
            setattr(self.expert_query.weight, 'sequence_parallel', True)

    def _compute_scores(self, input_n: torch.Tensor) -> torch.Tensor:
        """
        Compute raw cosine similarity scores using only Q and K (no V, no softmax).

        Args:
            input_n: Layer-normed input [N, H]

        Returns:
            scores: [N, E] cosine similarity per expert, bounded in [-1, 1]
        """
        N = input_n.shape[0]
        E = self.num_experts
        d2 = self.expert_attn_size

        # Encode to latent space
        H_e = self.expert_encoder(input_n).view(N, E, d2)  # [N, E, d2]

        # Compute per-expert queries
        Q = self.expert_query(H_e.reshape(N, E * d2)).view(N, E, d2)

        # Shared key
        K = self.key(input_n)  # [N, d2]

        # L2 normalize -> cosine similarity
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Raw cosine similarity (no scale, no softmax)
        # Q·K for unit vectors gives cosine similarity in [-1, 1]
        scores = (Q * K.unsqueeze(1)).sum(-1)  # [N, E]

        return scores

    def _compute_scores_and_latents(self, input_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute raw cosine similarity scores and latents (for filtering).

        Args:
            input_n: Layer-normed input [N, H]

        Returns:
            scores: [N, E] cosine similarity per expert, bounded in [-1, 1]
            H_e: [N, E, d2] latent representations per expert
        """
        N = input_n.shape[0]
        E = self.num_experts
        d2 = self.expert_attn_size

        # Encode to latent space
        H_e = self.expert_encoder(input_n).view(N, E, d2)  # [N, E, d2]

        # Compute per-expert queries
        Q = self.expert_query(H_e.reshape(N, E * d2)).view(N, E, d2)

        # Shared key
        K = self.key(input_n)  # [N, d2]

        # L2 normalize -> cosine similarity
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Raw cosine similarity (no scale, no softmax)
        scores = (Q * K.unsqueeze(1)).sum(-1)  # [N, E]

        return scores, H_e
    
    def _compute_filtered_inputs(self, H_e: torch.Tensor) -> torch.Tensor:
        """
        Project latent representations back to original dimension using transpose.
        """
        N, E, d2 = H_e.shape
        d1 = self.config.hidden_size

        # Get encoder weight: [E*d2, d1] -> [E, d2, d1]
        W = self.expert_encoder.weight.view(E, d2, d1)

        # Project back using transpose: x_tilde = W^T * H_e
        filtered_inputs = torch.einsum('edh,ned->neh', W, H_e)  # [N, E, H]

        return filtered_inputs

    def compute_orthonormality_loss(self) -> torch.Tensor:
        """Compute orthonormality regularization loss."""
        E = self.num_experts
        d2 = self.expert_attn_size
        d1 = self.config.hidden_size

        W = self.expert_encoder.weight.view(E, d2, d1)

        # W * W^T should be identity [E, d2, d2]
        WWT = torch.bmm(W, W.transpose(-1, -2))

        # Target: identity matrix
        I = torch.eye(d2, device=W.device, dtype=W.dtype).unsqueeze(0).expand(E, -1, -1)

        # Frobenius norm
        ortho_loss = ((WWT - I) ** 2).mean()

        return ortho_loss

    def gating(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing logits using EMA null state and Q,K attention.

        Returns:
            router_logits: [N, E]
            filtered_inputs: [N, E, H] or None
        """
        # Handle 3D input
        if input.dim() == 3:
            input = input.view(-1, input.shape[-1])

        N, H = input.shape
        orig_dtype = input.dtype

        # === 1. Update EMA during training ===
        if self.training:
            batch_mean = input.mean(dim=0, keepdim=True).detach()
            with torch.no_grad():
                self.global_null_mean.data = (1 - self.momentum) * self.global_null_mean.data + \
                                              self.momentum * batch_mean.to(self.global_null_mean.dtype)

        # === 2. Real Input: Log-Probs (+ Latents if filtering) ===
        real_n = F.layer_norm(input, (H,), eps=1e-5)

        filtered_inputs = None
        if self.use_filtered_inputs:
            scores_real, H_e_real = self._compute_scores_and_latents(real_n)
            # Reconstruct from latent
            reconstruction = self._compute_filtered_inputs(H_e_real)  # [N, E, H]
            # RESIDUAL STREAM: 50% Filtered + 50% Original
            filtered_inputs = 0.5 * reconstruction + 0.5 * input.unsqueeze(1)
        else:
            scores_real = self._compute_scores(real_n)

        # === 3. EMA Null Log-Probs ===
        null = self.global_null_mean.expand(N, H).to(orig_dtype)
        null_n = F.layer_norm(null, (H,), eps=1e-5)
        with torch.no_grad():
            scores_null = self._compute_scores(null_n)

        # === 4. Compute Routing Logits (Likelihood Ratio) ===
        router_logits = scores_real - scores_null

        # === 5. Finalize ===
        temp_clamped = torch.clamp(self.routing_temperature, min=0.1, max=10.0)
        router_logits = router_logits * temp_clamped
        router_logits = router_logits - router_logits.mean(-1, keepdim=True)
        router_logits = torch.clamp(router_logits, min=-10.0, max=10.0)

        return router_logits, filtered_inputs

    def apply_input_jitter(self, input: torch.Tensor) -> torch.Tensor:
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        return input

    def apply_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
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

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
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

    def apply_orthonormality_loss(self, activation: torch.Tensor) -> torch.Tensor:
        """Apply orthonormality regularization loss."""
        if self.use_ortho_reg and self.training:
            ortho_loss = self.compute_orthonormality_loss()
            save_to_aux_losses_tracker(
                "orthonormality_loss",
                ortho_loss,
                self.layer_number,
                self.config.num_layers,
            )
            activation = MoEAuxLossAutoScaler.apply(activation, ortho_loss * self.ortho_reg_coeff)
        return activation

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = logits.view(-1, self.config.num_moe_experts)
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            logits = gather_from_sequence_parallel_region(logits)

        if self.use_dynamic_routing and self.top_p < 1.0:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            top_p_mask = cumsum <= self.top_p
            top_p_mask[..., 0] = True
            allowed = torch.zeros_like(probs, dtype=torch.bool)
            allowed.scatter_(1, sorted_idx, top_p_mask)
            logits = logits.masked_fill(~allowed, float("-inf"))

        use_pre_softmax = (self.topk == 1)

        scores, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=use_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )

        if self.routing_type == "aux_loss" and self.training:
            scores_for_loss = torch.softmax(logits, dim=-1, dtype=torch.float32)
            scores = self.apply_load_balancing_loss(
                scores_for_loss,
                tokens_per_expert,
                activation=scores,
            )

        # Apply orthonormality loss if using filtered inputs
        if self.use_filtered_inputs:
            scores = self.apply_orthonormality_loss(scores)

        return scores, routing_map

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            scores: [N, E] routing scores
            routing_map: routing assignments
            filtered_inputs: [N, E, H] or None
        """
        self.hidden = input.shape[-1]
        input = self.apply_input_jitter(input)

        logits, filtered_inputs = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        return scores, routing_map, filtered_inputs

