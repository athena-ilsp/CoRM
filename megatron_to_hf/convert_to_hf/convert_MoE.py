import os, json, shutil, argparse
import torch
from safetensors.torch import save_file

MODEL_CONFIG = {
    "model_type": "standard_moe",
    "architectures": ["StandardMoEForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_standard_moe.StandardMoEConfig",
        "AutoModelForCausalLM": "modeling_standard_moe.StandardMoEForCausalLM"
    },

    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "intermediate_size": 4096,

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

# Fill in with the path to your MoE_model_scripts dir.
CUSTOM_MODEL_FILES_DIR = "CUSTOM_MODEL_FILES_DIR"

def load_legacy_pt(ckpt_path, dtype=torch.bfloat16):
    """Load a Megatron legacy checkpoint (model_optim_rng.pt) from mp_rank_00."""
    pt_file = os.path.join(ckpt_path, "mp_rank_00", "model_optim_rng.pt")
    ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
    raw = ckpt["model"]
    state = {}
    for k, v in raw.items():
        if "_extra_state" in k:
            continue
        if isinstance(v, torch.Tensor):
            state[k] = v.to(dtype)
    return state

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
    ap.add_argument("--tokenizer", default=None, help="optional tokenizer dir to copy files from")
    ap.add_argument("--model-files", default=CUSTOM_MODEL_FILES_DIR,
                    help="dir containing __init__.py, modeling_standard_moe.py, configuration_standard_moe.py")
    ap.add_argument("--num-experts", type=int, default=None, help="override num_local_experts")
    ap.add_argument("--topk", type=int, default=None, help="override num_experts_per_tok")
    ap.add_argument("--shared-expert-intermediate-size", type=int, default=None,
                    help="override shared_expert_intermediate_size (0 = no shared expert)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    meg = load_legacy_pt(args.input, dtype=torch.bfloat16)

    cfg = MODEL_CONFIG
    if args.num_experts is not None:
        cfg["num_local_experts"] = args.num_experts
    if args.topk is not None:
        cfg["num_experts_per_tok"] = args.topk
    if args.shared_expert_intermediate_size is not None:
        cfg["shared_expert_intermediate_size"] = args.shared_expert_intermediate_size
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

        # router - simple linear gate
        hf[f"{hp}.block_sparse_moe.router.gate.weight"] = meg[f"{mp}.mlp.router.weight"].clone()

        # experts (legacy format: per-expert tensors linear_fc1.weight0..N)
        for e in range(E):
            expert_fc1 = meg[f"{mp}.mlp.experts.linear_fc1.weight{e}"]  # [2I, H]
            expert_fc2 = meg[f"{mp}.mlp.experts.linear_fc2.weight{e}"]  # [H, I]
            gate_proj = expert_fc1[:I, :].contiguous()
            up_proj   = expert_fc1[I:, :].contiguous()
            down_proj = expert_fc2.contiguous()

            hf[f"{hp}.block_sparse_moe.experts.{e}.w1.weight"] = gate_proj
            hf[f"{hp}.block_sparse_moe.experts.{e}.w3.weight"] = up_proj
            hf[f"{hp}.block_sparse_moe.experts.{e}.w2.weight"] = down_proj

        # Shared expert (if present)
        shared_fc1_key = f"{mp}.mlp.shared_experts.linear_fc1.weight"
        if shared_fc1_key in meg:
            shared_fc1 = meg[shared_fc1_key]  # [2*shared_I, H]
            shared_fc2 = meg[f"{mp}.mlp.shared_experts.linear_fc2.weight"]  # [H, shared_I]
            shared_I = shared_fc1.shape[0] // 2
            hf[f"{hp}.block_sparse_moe.shared_expert.w1.weight"] = shared_fc1[:shared_I, :].contiguous()
            hf[f"{hp}.block_sparse_moe.shared_expert.w3.weight"] = shared_fc1[shared_I:, :].contiguous()
            hf[f"{hp}.block_sparse_moe.shared_expert.w2.weight"] = shared_fc2.contiguous()

    # write config.json
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # save weights
    save_file({k: v.to(torch.bfloat16) for k, v in hf.items()},
              os.path.join(args.output, "model.safetensors"))

    # copy custom model files
    custom_files = ["__init__.py", "modeling_standard_moe.py", "configuration_standard_moe.py"]
    for fn in custom_files:
        src = os.path.join(args.model_files, fn)
        if os.path.exists(src):
            shutil.copy(src, args.output)
            print(f"  Copied {fn}")
        else:
            print(f"  WARNING: {fn} not found at {src}")

    # copy tokenizer files if provided
    if args.tokenizer:
        for fn in [
            "tokenizer.json","tokenizer_config.json","special_tokens_map.json",
            "vocab.json","merges.txt","vocab.bpe","encoder.json"
        ]:
            src = os.path.join(args.tokenizer, fn)
            if os.path.exists(src):
                shutil.copy(src, args.output)

    print("✅ Exported HF custom MoE checkpoint to:", args.output)

if __name__ == "__main__":
    main()