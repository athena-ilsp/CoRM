import os, json, shutil, argparse
import sys
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from safetensors.torch import save_file

# Fill in with the path to your CoRM_model_scripts dir and MoE_model_scripts dir
# (see megatron_to_hf/CoRM_model_scripts and megatron_to_hf/MoE_model_scripts).
CUSTOM_MODEL_FILES_DIR = "CUSTOM_MODEL_FILES_DIR"
MOE_MODEL_FILES_DIR = "MOE_MODEL_FILES_DIR"

# Add ReMoE path for loading legacy checkpoints with Megatron classes
REMOE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REMOE_PATH not in sys.path:
    sys.path.insert(0, REMOE_PATH)

def get_model_config(router_type="corm", **kwargs):
    """Get model config with router type and optional overrides."""

    if router_type == "xmoe":
        config = {
            "model_type": "standard_moe",
            "architectures": ["XMoEForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_standard_moe.StandardMoEConfig",
                "AutoModelForCausalLM": "modeling_standard_moe.XMoEForCausalLM"
            },
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 3072,
            "num_local_experts": 8,
            "num_experts_per_tok": 1,
            "shared_expert_intermediate_size": 0,
            "max_position_embeddings": 1024,
            "vocab_size": 51200,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16",
            "tie_word_embeddings": False,
            "xmoe_expert_dim": 4,
        }
        config.update(kwargs)
        return config

    if router_type == "relu":
        config = {
            "model_type": "standard_moe",
            "architectures": ["StandardMoEForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_standard_moe.StandardMoEConfig",
                "AutoModelForCausalLM": "modeling_standard_moe.StandardMoEForCausalLM"
            },
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 3072,
            "num_local_experts": 8,
            "num_experts_per_tok": 1,
            "shared_expert_intermediate_size": 0,
            "max_position_embeddings": 1024,
            "vocab_size": 51200,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16",
            "tie_word_embeddings": False,
        }
        config.update(kwargs)
        return config

    # CoRM (Contrastive Routing Mixture-of-Experts) — the only RIM-family
    # router type this pipeline exports.
    config = {
        "model_type": "corm",
        "architectures": ["CoRMForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_corm.CoRMConfig",
            "AutoModelForCausalLM": "modeling_corm.CoRMForCausalLM"
        },

        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "intermediate_size": 3072,

        "num_local_experts": 8,  # Will be auto-detected from checkpoint
        "num_experts_per_tok": 1,  # Read from checkpoint args (moe_router_topk)
        # Dynamic top-p routing is NOT implemented in the HF modeling code; the
        # converter refuses checkpoints trained with it (see main()).
        "use_dynamic_routing": False,
        "top_p": 1.0,
        "expert_attn_size": 64,  # Auto-detected from router key.weight
        "max_position_embeddings": 1024,
        "vocab_size": 51200,

        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "routing_temperature_init": 1.0,

        "router_type": router_type,
        "router_background_momentum": 0.01,  # Read from checkpoint args (moe_ema_null_momentum)
        "moe_router_use_l2_normalize": True,

        # The corm router has no shared expert.
        "has_shared_expert": False,
        "learn_output_weights": False,
        "fixed_output_weights": False,

        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
    }

    # Apply any overrides from kwargs
    config.update(kwargs)

    return config

def load_legacy_checkpoint(ckpt_path, dtype=torch.bfloat16):
    """Load legacy Megatron checkpoint format (model_optim_rng.pt)."""
    import glob

    # Find model_optim_rng.pt files
    pattern = os.path.join(ckpt_path, "mp_rank_*", "model_optim_rng.pt")
    ckpt_files = sorted(glob.glob(pattern))

    if not ckpt_files:
        raise FileNotFoundError(f"No model_optim_rng.pt files found in {ckpt_path}")

    print(f"Loading legacy checkpoint from {len(ckpt_files)} rank(s)...")

    # Load from rank 0 (weights_only=False for Megatron checkpoints)
    ckpt = torch.load(ckpt_files[0], map_location='cpu', weights_only=False)

    # Extract model state dict
    if 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt

    # Convert to target dtype
    state = {k: v.to(dtype) if torch.is_tensor(v) else v for k, v in state.items()}

    print(f"Loaded {len(state)} keys from checkpoint")
    return state

def load_dcp(ckpt_path, dtype=torch.bfloat16):
    """Load Distributed Checkpoint (DCP) format."""
    reader = FileSystemReader(ckpt_path)
    md = reader.read_metadata()
    state = {}
    for k, meta in md.state_dict_metadata.items():
        if (k.startswith("optimizer.") or "_extra_state" in k or k == "rng_state" or "/shard_" in k):
            continue
        if hasattr(meta, "size"):
            state[k] = torch.zeros(list(meta.size), dtype=dtype)
    dcp.load(state_dict=state, storage_reader=reader)
    return state

def load_checkpoint(ckpt_path, dtype=torch.bfloat16):
    """Auto-detect and load checkpoint format."""
    # Check if DCP format (.metadata file exists)
    if os.path.exists(os.path.join(ckpt_path, ".metadata")):
        print("Detected DCP format")
        return load_dcp(ckpt_path, dtype)
    # Check if legacy format (mp_rank_*/model_optim_rng.pt exists)
    elif os.path.exists(os.path.join(ckpt_path, "mp_rank_00", "model_optim_rng.pt")):
        print("Detected legacy Megatron format")
        return load_legacy_checkpoint(ckpt_path, dtype)
    else:
        raise ValueError(f"Unknown checkpoint format in {ckpt_path}")

def convert_qkv_megatron_to_hf(qkv_weight, num_kv_heads, num_q_per_kv, head_dim, hidden_size):
    group_size = (num_q_per_kv + 2) * head_dim
    qkv_grouped = qkv_weight.view(num_kv_heads, group_size, hidden_size)

    q_span = num_q_per_kv * head_dim
    k_span = head_dim
    q_groups = qkv_grouped[:, :q_span, :]
    k_groups = qkv_grouped[:, q_span:q_span+k_span, :]
    v_groups = qkv_grouped[:, q_span+k_span:, :]

    q = q_groups.reshape(num_kv_heads * q_span, hidden_size)
    k = k_groups.reshape(num_kv_heads * k_span, hidden_size)
    v = v_groups.reshape(num_kv_heads * k_span, hidden_size)
    return q, k, v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Megatron DCP checkpoint dir")
    ap.add_argument("--output", required=True, help="HF output dir")

    ap.add_argument("--router-type", default="corm",
                    choices=["corm", "relu", "xmoe"],
                    help="Router type to use")

    ap.add_argument("--tokenizer", default=None, help="optional tokenizer dir to copy files from")
    ap.add_argument("--model-files", default=CUSTOM_MODEL_FILES_DIR,
                    help="dir containing modeling files")

    # === Model dimension overrides ===
    ap.add_argument("--expert-attn-size", type=int, default=None,
                    help="Router attention dimension (auto-detected from checkpoint if not set)")
    ap.add_argument("--num-attention-heads", type=int, default=None,
                    help="Number of attention heads (auto-detected if not set)")
    ap.add_argument("--num-key-value-heads", type=int, default=None,
                    help="Number of KV heads (auto-detected if not set)")
    ap.add_argument("--topk", type=int, default=None,
                    help="Number of experts per token (auto-detected from checkpoint if not set)")

    args = ap.parse_args()

    # Auto-select model files for relu/xmoe if not explicitly overridden
    if args.router_type in ("relu", "xmoe") and args.model_files == CUSTOM_MODEL_FILES_DIR:
        args.model_files = MOE_MODEL_FILES_DIR

    print(f"Converting checkpoint with router type: {args.router_type}")

    os.makedirs(args.output, exist_ok=True)
    meg = load_checkpoint(args.input, dtype=torch.bfloat16)

    # Auto-detect number of experts from checkpoint
    num_experts = 0
    for i in range(100):  # Check up to 100 experts
        if f"decoder.layers.0.mlp.experts.linear_fc1.weight{i}" in meg:
            num_experts = i + 1
        else:
            break

    if num_experts == 0:
        # Try stacked format
        stacked_key = "decoder.layers.0.mlp.experts.experts.linear_fc1.weight"
        if stacked_key in meg:
            num_experts = meg[stacked_key].shape[0]

    print(f"Auto-detected {num_experts} experts from checkpoint")

    # Auto-detect model dimensions from checkpoint tensors
    embed_weight = meg["embedding.word_embeddings.weight"]
    hidden_size = embed_weight.shape[1]
    num_hidden_layers = sum(
        1 for k in meg if k.startswith("decoder.layers.") and k.endswith(".self_attention.linear_qkv.weight")
    )
    qkv_weight = meg["decoder.layers.0.self_attention.linear_qkv.weight"]  # [nk*(q_per_kv+2)*head_dim, H]
    # intermediate_size from expert fc1 (fused gate+up = 2*I)
    fc1_key = next((k for k in meg if "mlp.experts.linear_fc1.weight0" in k or
                    "mlp.experts.experts.linear_fc1.weight" in k), None)
    if fc1_key:
        fc1_shape = meg[fc1_key].shape
        intermediate_size = (fc1_shape[0] if "weight0" in fc1_key else fc1_shape[1]) // 2
    else:
        intermediate_size = hidden_size * 4  # fallback

    # Detect num_attention_heads and num_key_value_heads from QKV shape
    # QKV rows = nk * (q_per_kv + 2) * head_dim, where head_dim = hidden_size / num_heads
    # Prefer larger num_heads (smaller head_dim), standard values are 64 or 128.
    qkv_rows = qkv_weight.shape[0]
    num_attention_heads, num_key_value_heads = None, None
    for nh_try in sorted([4, 8, 12, 16, 32], reverse=False):  # prefer head_dim=64 (fewer heads first)
        head_dim_try = hidden_size // nh_try
        if hidden_size % nh_try != 0 or head_dim_try > 128 or head_dim_try < 32:
            continue
        for nk_try in [4, 2, 1, 8]:
            if nh_try % nk_try != 0:
                continue
            q_per_kv_try = nh_try // nk_try
            expected_rows = nk_try * (q_per_kv_try + 2) * head_dim_try
            if expected_rows == qkv_rows:
                num_attention_heads = nh_try
                num_key_value_heads = nk_try
                break
        if num_attention_heads is not None:
            break
    if num_attention_heads is None:
        raise ValueError(f"Could not infer attention head config from QKV shape {qkv_weight.shape} with hidden_size={hidden_size}. "
                         f"Use --num-attention-heads and --num-key-value-heads to specify explicitly.")

    # Allow CLI overrides
    if args.num_attention_heads is not None:
        num_attention_heads = args.num_attention_heads
    if args.num_key_value_heads is not None:
        num_key_value_heads = args.num_key_value_heads

    print(f"Auto-detected: hidden_size={hidden_size}, num_layers={num_hidden_layers}, "
          f"num_heads={num_attention_heads}, num_kv_heads={num_key_value_heads}, "
          f"intermediate_size={intermediate_size}")

    # Auto-detect expert_attn_size (d2) from router key projection
    expert_attn_size = args.expert_attn_size
    if expert_attn_size is None:
        key_weight_key = "decoder.layers.0.mlp.router.key.weight"
        if key_weight_key in meg:
            expert_attn_size = meg[key_weight_key].shape[0]
            print(f"Auto-detected expert_attn_size = {expert_attn_size} from router key.weight")
        else:
            expert_attn_size = 64  # fallback default
            print(f"Warning: Could not auto-detect expert_attn_size, using default {expert_attn_size}")
    else:
        print(f"Using provided expert_attn_size = {expert_attn_size}")

    # Auto-detect topk and relu_routing from checkpoint args (common.pt for DCP, ckpt itself for legacy)
    ckpt_args = None
    try:
        _common = torch.load(os.path.join(args.input, "common.pt"), map_location="cpu", weights_only=False)
        ckpt_args = _common.get("args", None)
        print(f"Loaded args from common.pt")
    except Exception:
        try:
            import glob
            legacy_files = sorted(glob.glob(os.path.join(args.input, "mp_rank_*", "model_optim_rng.pt")))
            if legacy_files:
                _ckpt = torch.load(legacy_files[0], map_location="cpu", weights_only=False)
                ckpt_args = _ckpt.get("args", None)
                print(f"Loaded args from legacy ckpt")
        except Exception:
            pass

    topk = args.topk
    if topk is None:
        topk = getattr(ckpt_args, "moe_router_topk", 1) if ckpt_args else 1
        print(f"Auto-detected topk = {topk}")

    # For relu router-type, switch to true ReLUMoE if training used moe_relu_routing=True
    ckpt_relu_routing = getattr(ckpt_args, "moe_relu_routing", False) if ckpt_args else False
    ckpt_router_type = getattr(ckpt_args, "moe_router_type", "topk") if ckpt_args else "topk"
    use_relu_moe = (args.router_type == "relu" and ckpt_relu_routing and ckpt_router_type == "relu")
    if args.router_type == "relu":
        print(f"  ckpt moe_router_type={ckpt_router_type}, moe_relu_routing={ckpt_relu_routing} → HF class: {'ReLUMoEForCausalLM' if use_relu_moe else 'StandardMoEForCausalLM'}")

    # Detect L2-normalize flag from checkpoint (default True — matches Megatron default)
    ckpt_use_l2_normalize = getattr(ckpt_args, "moe_router_use_l2_normalize", True) if ckpt_args else True
    if args.router_type == "corm":
        print(f"  ckpt moe_router_use_l2_normalize={ckpt_use_l2_normalize}")

    # Build config with appropriate settings
    # The QKV shape alone is ambiguous (e.g. H=1024: 16 heads/4 groups and 8 heads/2 groups
    # give the same rows), so prefer the head counts recorded in the training args.
    if ckpt_args is not None:
        if args.num_attention_heads is None and getattr(ckpt_args, "num_attention_heads", None):
            num_attention_heads = ckpt_args.num_attention_heads
        if args.num_key_value_heads is None and getattr(ckpt_args, "group_query_attention", False):
            num_key_value_heads = ckpt_args.num_query_groups
        print(f"Using attention heads from checkpoint args: num_heads={num_attention_heads}, "
              f"num_kv_heads={num_key_value_heads}")
    head_dim = hidden_size // num_attention_heads
    expected_rows = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    if expected_rows != qkv_weight.shape[0]:
        raise ValueError(f"QKV shape {tuple(qkv_weight.shape)} inconsistent with num_heads={num_attention_heads}, "
                         f"num_kv_heads={num_key_value_heads}")

    config_kwargs = {
        "num_local_experts": num_experts,
        "num_experts_per_tok": topk,
        "expert_attn_size": expert_attn_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": intermediate_size,
    }

    # Shape/numerics that must match training exactly
    config_kwargs["vocab_size"] = embed_weight.shape[0]
    if ckpt_args is not None:
        config_kwargs["rope_theta"] = float(getattr(ckpt_args, "rotary_base", 1000000))
        config_kwargs["rms_norm_eps"] = float(getattr(ckpt_args, "norm_epsilon", 1e-5))
        config_kwargs["max_position_embeddings"] = int(getattr(ckpt_args, "max_position_embeddings", 1024))

    if args.router_type == "corm":
        # The HF CoRM implementation only supports fixed top-k routing.
        if ckpt_args is not None and getattr(ckpt_args, "use_dynamic_routing", False) \
                and getattr(ckpt_args, "top_p", 1.0) < 1.0:
            raise ValueError("Checkpoint was trained with dynamic top-p routing, "
                             "which modeling_corm.py does not implement.")
        config_kwargs["use_dynamic_routing"] = False
        config_kwargs["top_p"] = 1.0
        if ckpt_args is not None:
            config_kwargs["router_background_momentum"] = float(getattr(
                ckpt_args, "moe_ema_null_momentum",
                getattr(ckpt_args, "router_background_momentum", 0.01)))
        # The corm router has no shared expert
        config_kwargs["has_shared_expert"] = False
        config_kwargs["learn_output_weights"] = False
        config_kwargs["fixed_output_weights"] = False

    # Propagate L2-normalize flag (default True) to HF config
    config_kwargs["moe_router_use_l2_normalize"] = ckpt_use_l2_normalize

    cfg = get_model_config(args.router_type, **config_kwargs)

    # Switch to ReLUMoE architecture if training used true ReLU routing
    if use_relu_moe:
        cfg["architectures"] = ["ReLUMoEForCausalLM"]
        cfg["auto_map"]["AutoModelForCausalLM"] = "modeling_standard_moe.ReLUMoEForCausalLM"

    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    nh = cfg["num_attention_heads"]
    nk = cfg["num_key_value_heads"]
    head_dim = H // nh
    q_per_kv = nh // nk
    E = cfg["num_local_experts"]
    I = cfg["intermediate_size"]

    hf = {}

    # embeddings / lm head / final norm
    hf["model.embed_tokens.weight"] = meg["embedding.word_embeddings.weight"].clone()
    hf["lm_head.weight"] = meg["output_layer.weight"].clone()
    hf["model.norm.weight"] = meg["decoder.final_layernorm.weight"].clone()

    for l in range(L):
        mp = f"decoder.layers.{l}"
        hp = f"model.layers.{l}"

        # attention
        qkv = meg[f"{mp}.self_attention.linear_qkv.weight"]
        q, k, v = convert_qkv_megatron_to_hf(qkv, nk, q_per_kv, head_dim, H)

        hf[f"{hp}.self_attn.q_proj.weight"] = q.contiguous()
        hf[f"{hp}.self_attn.k_proj.weight"] = k.contiguous()
        hf[f"{hp}.self_attn.v_proj.weight"] = v.contiguous()
        hf[f"{hp}.self_attn.o_proj.weight"] = meg[f"{mp}.self_attention.linear_proj.weight"].clone()

        # norms
        hf[f"{hp}.input_layernorm.weight"] = meg[f"{mp}.self_attention.linear_qkv.layer_norm_weight"].clone()
        hf[f"{hp}.post_attention_layernorm.weight"] = meg[f"{mp}.pre_mlp_layernorm.weight"].clone()

        # experts - handle both stacked and individual weight formats
        # relu/xmoe use block_sparse_moe.experts, corm uses moe.experts
        expert_prefix = "block_sparse_moe" if args.router_type in ("relu", "xmoe") else "moe"
        stacked_fc1_key = f"{mp}.mlp.experts.experts.linear_fc1.weight"
        individual_fc1_key = f"{mp}.mlp.experts.linear_fc1.weight0"

        if stacked_fc1_key in meg:
            fc1 = meg[stacked_fc1_key]
            fc2 = meg[f"{mp}.mlp.experts.experts.linear_fc2.weight"]
            for e in range(E):
                expert_fc1 = fc1[e]
                hf[f"{hp}.{expert_prefix}.experts.{e}.w1.weight"] = expert_fc1[:I, :].contiguous()
                hf[f"{hp}.{expert_prefix}.experts.{e}.w3.weight"] = expert_fc1[I:, :].contiguous()
                hf[f"{hp}.{expert_prefix}.experts.{e}.w2.weight"] = fc2[e].contiguous()

        elif individual_fc1_key in meg:
            for e in range(E):
                expert_fc1 = meg[f"{mp}.mlp.experts.linear_fc1.weight{e}"]
                expert_fc2 = meg[f"{mp}.mlp.experts.linear_fc2.weight{e}"]
                hf[f"{hp}.{expert_prefix}.experts.{e}.w1.weight"] = expert_fc1[:I, :].contiguous()
                hf[f"{hp}.{expert_prefix}.experts.{e}.w3.weight"] = expert_fc1[I:, :].contiguous()
                hf[f"{hp}.{expert_prefix}.experts.{e}.w2.weight"] = expert_fc2.contiguous()
        else:
            raise KeyError(f"Could not find expert weights in expected format for layer {l}")

        # ============================================================
        # Router params (Handled conditionally based on type)
        # ============================================================

        # Debug: Print available router keys for first layer
        if l == 0:
            print("\n--- DEBUG: Available Router Keys in Checkpoint ---")
            router_keys = [k for k in meg.keys() if "router" in k and f"layers.{l}" in k]
            for k in sorted(router_keys):
                print(f"  {k}")
            print("--------------------------------------------------\n")

        # --- CASE: relu router (ReMoE) ---
        if args.router_type == "relu":
            # Simple linear gate: weight [E, H]
            hf[f"{hp}.block_sparse_moe.router.gate.weight"] = meg[f"{mp}.mlp.router.weight"].clone()

        # --- CASE: xmoe router (X-MoE) ---
        elif args.router_type == "xmoe":
            hf[f"{hp}.block_sparse_moe.router.proj_weight"] = meg[f"{mp}.mlp.router.proj_weight"].clone()
            hf[f"{hp}.block_sparse_moe.router.expert_embeddings"] = meg[f"{mp}.mlp.router.expert_embeddings"].clone()
            hf[f"{hp}.block_sparse_moe.router.temperature"] = meg[f"{mp}.mlp.router.temperature"].clone()

        # --- CASE: corm router ---
        else:
            hf[f"{hp}.moe.router.key.weight"] = meg[f"{mp}.mlp.router.key.weight"].clone()
            hf[f"{hp}.moe.router.expert_queries.weight"] = meg[f"{mp}.mlp.router.expert_queries.weight"].clone()
            hf[f"{hp}.moe.router.routing_temperature"] = meg[f"{mp}.mlp.router.routing_temperature"].clone()
            null_key = f"{mp}.mlp.router.global_null_mean"
            if null_key in meg:
                hf[f"{hp}.moe.router.global_null_mean"] = meg[null_key].clone()
                hf[f"{hp}.moe.router.ema_initialized"] = torch.tensor(True)
                if l == 0:
                    print(f"  global_null_mean found in checkpoint (norm={meg[null_key].norm():.4f})")
            else:
                raise KeyError(f"{null_key} not found: the CoRM EMA null state is required for "
                               f"correct routing and is saved by the CoRM Megatron router.")

        # ============================================================
        # Shared expert (corm has none; kept for relu/xmoe compatibility)
        # ============================================================
        shared_fc1_key = f"{mp}.mlp.shared_experts.linear_fc1.weight"
        if shared_fc1_key in meg:
            shared_fc1 = meg[shared_fc1_key]
            shared_fc2 = meg[f"{mp}.mlp.shared_experts.linear_fc2.weight"]

            hf[f"{hp}.moe.shared_expert.w1.weight"] = shared_fc1[:I, :].contiguous()
            hf[f"{hp}.moe.shared_expert.w3.weight"] = shared_fc1[I:, :].contiguous()
            hf[f"{hp}.moe.shared_expert.w2.weight"] = shared_fc2.contiguous()
            print(f"  Layer {l}: Converted shared expert")

    # write config.json
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # save weights
    save_file({k: v.to(torch.bfloat16) for k, v in hf.items()},
              os.path.join(args.output, "model.safetensors"))

    # copy custom model files
    if args.router_type in ("relu", "xmoe"):
        custom_files = ["__init__.py", "modeling_standard_moe.py", "configuration_standard_moe.py"]
    else:
        custom_files = ["__init__.py", "modeling_corm.py", "configuration_corm.py"]
    for fn in custom_files:
        src = os.path.join(args.model_files, fn)
        if os.path.exists(src):
            shutil.copy(src, args.output)
            print(f"  Copied {fn}")
        else:
            print(f"  WARNING: {fn} not found at {src}")

    # copy tokenizer files
    if args.tokenizer:
        for fn in [
            "tokenizer.json","tokenizer_config.json","special_tokens_map.json",
            "vocab.json","merges.txt","vocab.bpe","encoder.json"
        ]:
            src = os.path.join(args.tokenizer, fn)
            if os.path.exists(src):
                shutil.copy(src, args.output)

    print(f"✅ Exported HF checkpoint with router type '{args.router_type}' to: {args.output}")

if __name__ == "__main__":
    main()
