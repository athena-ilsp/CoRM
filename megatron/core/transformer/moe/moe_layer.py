# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple

import torch.nn as nn
import torch
import math

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import (
    TopKRouter,
    XMoERouter,
    ReLURouter,
    CoRM,
)
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


# ===================================================================
# Router Registry
# ===================================================================
ROUTER_REGISTRY = {
    'topk': TopKRouter,
    'xmoe': XMoERouter,
    'relu': ReLURouter,
    'corm': CoRM,
}


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None):
        super().__init__(config=config, layer_number=layer_number)
        self.submodules = submodules

        # Router type
        self.router_type = getattr(config, 'moe_router_type', 'topk')

        # Shared expert combination mode: 'add', 'learnable', 'fixed'
        self.combine_mode = getattr(config, 'moe_shared_expert_combine', 'add')

        # Logging
        self.moe_log_interval = getattr(config, 'moe_log_interval', 1000)
        self.num_layers = config.num_layers

        self.moe_layer_recompute = config.moe_layer_recompute

        # === Shared expert output combination ===
        if self.combine_mode == 'fixed':
            shared_expert_weight = getattr(config, 'shared_expert_weight', None)
            routed_expert_weight = getattr(config, 'routed_expert_weight', None)
            assert shared_expert_weight is not None and routed_expert_weight is not None, \
                "Must set --shared-expert-weight and --routed-expert-weight when using --moe-shared-expert-combine fixed"
            self.shared_expert_weight = shared_expert_weight
            self.routed_expert_weight = routed_expert_weight
            self.register_buffer('shared_logit', torch.tensor([0.0]))
            print(f"[Layer {layer_number}] Combine=fixed: "
                  f"shared={self.shared_expert_weight}, routed={self.routed_expert_weight}")
        elif self.combine_mode == 'learnable':
            self.shared_logit = nn.Parameter(torch.zeros(1))
            print(f"[Layer {layer_number}] Combine=learnable (init logit=0.0)")
        else:
            # 'add' mode: output = shared + routed
            self.register_buffer('shared_logit', torch.tensor([0.0]))
            if getattr(config, 'moe_shared_expert_intermediate_size', None) is not None:
                print(f"[Layer {layer_number}] Combine=add (output = shared + routed)")

        # === Initialize router from registry ===
        if self.router_type not in ROUTER_REGISTRY:
            raise ValueError(
                f"Unknown router type '{self.router_type}'. "
                f"Available: {list(ROUTER_REGISTRY.keys())}"
            )
        router_cls = ROUTER_REGISTRY[self.router_type]
        self.router = router_cls(config=self.config)
        print(f"[Layer {layer_number}] Router: {self.router_type} ({router_cls.__name__})")

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)

    # ===================================================================
    # Utilities
    # ===================================================================

    def set_iteration(self, iteration: int):
        self.current_iteration = iteration

    def _get_current_iteration(self) -> int:
        try:
            from megatron.training import get_args
            args = get_args()
            global_batch_size = getattr(args, 'global_batch_size', None)
            if global_batch_size is None:
                dp_size = getattr(args, 'data_parallel_size', 1)
                micro_bs = getattr(args, 'micro_batch_size', 1)
                num_micro = getattr(args, 'num_microbatches', 1)
                global_batch_size = dp_size * micro_bs * num_micro
            if global_batch_size > 0:
                return args.consumed_train_samples // global_batch_size
            return 0
        except:
            return 0

    def _combine_outputs(self, shared_output, routed_output):
        """Combine shared and routed expert outputs based on combine_mode."""
        if self.combine_mode == 'add':
            return shared_output + routed_output
        elif self.combine_mode == 'learnable':
            w = torch.sigmoid(self.shared_logit)
            return w * shared_output + (1.0 - w) * routed_output
        elif self.combine_mode == 'fixed':
            return self.shared_expert_weight * shared_output + self.routed_expert_weight * routed_output
        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

    # ===================================================================
    # Forward
    # ===================================================================

    def forward(self, hidden_states: torch.Tensor):

        def custom_forward(hidden_states):
            # --- 1. Shared expert (if present) ---
            shared_output = None
            if self.use_shared_expert and not self.shared_expert_overlap:
                shared_output = self.shared_experts(hidden_states)

            # --- 2. Routing ---
            probs, routing_map = self.router(hidden_states)

            # --- 3. Dispatch to experts ---
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )

            # --- 4. Expert computation ---
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            routed_output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)

            # --- 5. Combine with shared expert ---
            if shared_output is not None:
                output = self._combine_outputs(shared_output, routed_output)
            else:
                output = routed_output

            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias
