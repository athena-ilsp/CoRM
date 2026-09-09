#---------------------------------
#       Logging
#---------------------------------

from megatron.training import get_args
from megatron.core import mpu
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class MoEStatsTracker:
    """Minimal tracker for MoE routing statistics."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.stats: Dict[int, Dict[str, list]] = {}  # layer_num -> {stat_name -> [values]}
        self.log_interval = 100
    
    def set_log_interval(self, interval: int):
        self.log_interval = interval
    
    def reset(self):
        self.stats = {}
    
    def record(self, layer_num: int, log_ratio_std: float, shared_weight: float, router_type: float):
        """Record stats for a layer."""
        if layer_num not in self.stats:
            self.stats[layer_num] = {
                'log_ratio_std': [],
                'shared_weight': [],
                'router_type': [],
            }
        self.stats[layer_num]['log_ratio_std'].append(log_ratio_std)
        self.stats[layer_num]['shared_weight'].append(shared_weight)
        self.stats[layer_num]['router_type'].append(router_type)
    
    def get_and_reset(self) -> Optional[str]:
        """Get formatted log string and reset stats."""
        if not self.stats:
            return None
        
        lines = ["[MoE Routing]"]
        
        for layer_num in sorted(self.stats.keys()):
            layer_stats = self.stats[layer_num]
            
            # Average over accumulated steps
            log_ratio_std = sum(layer_stats['log_ratio_std']) / len(layer_stats['log_ratio_std'])
            shared_weight = sum(layer_stats['shared_weight']) / len(layer_stats['shared_weight'])
            router_type = layer_stats['router_type'][-1]  # Take last value (0=TopK, 1=RIM)
            
            router_name = "RIM" if router_type > 0.5 else "TopK"
            lines.append(
                f"  L{layer_num}: router={router_name} | "
                f"log_ratio_std={log_ratio_std:.4f} | "
                f"shared_w={shared_weight:.4f}"
            )
        
        self.reset()
        return "\n".join(lines)


def get_moe_stats_tracker() -> MoEStatsTracker:
    """Get global MoE stats tracker."""
    return MoEStatsTracker()


# ============================================================================
# MEGATRON-LM INTEGRATION
# ============================================================================

def update_moe_routing_and_log(model, step: int):
    """
    Update routing step and log stats.
    Call this once per training step.
    
    Args:
        model: The model (or list of model chunks for pipeline parallel)
        step: Current training iteration
    """
    # Update step in all MoE layers
    if isinstance(model, list):
        for m in model:
            update_moe_routing_and_log(m, step)
        return
    
    for module in model.modules():
        if isinstance(module, MoELayer):
            module.update_routing_step(step)
    
    # Check if we should log (only on rank 0)
    tracker = get_moe_stats_tracker()
    
    if step > 0 and step % tracker.log_interval == 0:
        # Only log on data parallel rank 0 to avoid duplicate logs
        if mpu.get_data_parallel_rank() == 0:
            log_str = tracker.get_and_reset()
            if log_str:
                print_rank_0(log_str)


def print_rank_0(message: str):
    """Print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def setup_moe_logging(log_interval: int = 100):
    """
    Setup MoE logging with specified interval.
    Call this during training initialization.
    
    Args:
        log_interval: Log every N steps
    """
    tracker = get_moe_stats_tracker()
    tracker.set_log_interval(log_interval)
    print_rank_0(f"[MoE] Logging enabled every {log_interval} steps")