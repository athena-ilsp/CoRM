"""Convert a Megatron dense Llama-style DCP checkpoint to HuggingFace format.

Dense checkpoints use stacked tensors where all L layers are concatenated:
    decoder.layers.self_attention.linear_qkv.weight: [L, qkv_rows, H]
    decoder.layers.mlp.linear_fc1.weight:            [L, 2I, H]
    ...
"""

import os, json, shutil, argparse, sys
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from safetensors.torch import save_file

# Fill in with the path to your Dense_model_scripts dir.
CUSTOM_MODEL_FILES_DIR = "CUSTOM_MODEL_FILES_DIR"

REMOE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REMOE_PATH not in sys.path:
    sys.path.insert(0, REMOE_PATH)


def load_dcp(ckpt_path, dtype=torch.bfloat16):
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


def convert_qkv_megatron_to_hf(qkv_weight, num_kv_heads, num_q_per_kv, head_dim, hidden_size):
    group_size = (num_q_per_kv + 2) * head_dim
    qkv_grouped = qkv_weight.view(num_kv_heads, group_size, hidden_size)
    q_span = num_q_per_kv * head_dim
    k_span = head_dim
    q = qkv_grouped[:, :q_span, :].reshape(num_kv_heads * q_span, hidden_size)
    k = qkv_grouped[:, q_span:q_span+k_span, :].reshape(num_kv_heads * k_span, hidden_size)
    v = qkv_grouped[:, q_span+k_span:, :].reshape(num_kv_heads * k_span, hidden_size)
    return q, k, v


def read_heads_from_common(ckpt_path):
    """Read num_attention_heads and num_query_groups from Megatron common.pt."""
    common_pt = os.path.join(ckpt_path, "common.pt")
    if not os.path.exists(common_pt):
        return None, None
    try:
        common = torch.load(common_pt, map_location="cpu", weights_only=False)
        a = common.get("args")
        if a is None:
            return None, None
        nh = getattr(a, "num_attention_heads", None)
        nk = getattr(a, "num_query_groups", None) or getattr(a, "num_key_value_heads", None) or nh
        return nh, nk
    except Exception:
        return None, None


def detect_qkv_heads(qkv_rows, hidden_size, override_nh=None, override_nk=None):
    if override_nh is not None and override_nk is not None:
        return override_nh, override_nk
    # Prefer head_dim=64 (larger nh) to avoid wrong picks when multiple configs match
    for nh in [32, 16, 12, 8, 4]:
        if hidden_size % nh != 0:
            continue
        head_dim = hidden_size // nh
        if head_dim < 32 or head_dim > 128:
            continue
        for nk in [4, 2, 1, 8]:
            if nh % nk != 0:
                continue
            q_per_kv = nh // nk
            if nk * (q_per_kv + 2) * head_dim == qkv_rows:
                return nh, nk
    raise ValueError(f"Cannot infer attention heads from qkv_rows={qkv_rows}, hidden={hidden_size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--model-files", default=CUSTOM_MODEL_FILES_DIR)
    ap.add_argument("--num-attention-heads", type=int, default=None)
    ap.add_argument("--num-key-value-heads", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading DCP checkpoint from {args.input}")
    meg = load_dcp(args.input, dtype=torch.bfloat16)

    embed = meg["embedding.word_embeddings.weight"]
    H = embed.shape[1]

    qkv_stacked = meg["decoder.layers.self_attention.linear_qkv.weight"]  # [L, qkv_rows, H]
    L, qkv_rows, _ = qkv_stacked.shape

    # Read heads from common.pt first (authoritative), fall back to detection
    nh_common, nk_common = read_heads_from_common(args.input)
    nh_override = args.num_attention_heads or nh_common
    nk_override = args.num_key_value_heads or nk_common
    nh, nk = detect_qkv_heads(qkv_rows, H, nh_override, nk_override)
    head_dim = H // nh
    q_per_kv = nh // nk

    fc1_stacked = meg["decoder.layers.mlp.linear_fc1.weight"]  # [L, 2I, H]
    I = fc1_stacked.shape[1] // 2

    print(f"Detected: hidden_size={H}, num_layers={L}, num_heads={nh}, num_kv_heads={nk}, intermediate_size={I}")

    cfg = {
        "model_type": "dense_llama",
        "architectures": ["DenseLlamaForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_dense_llama.DenseLlamaConfig",
            "AutoModelForCausalLM": "modeling_dense_llama.DenseLlamaForCausalLM",
        },
        "hidden_size": H,
        "num_hidden_layers": L,
        "num_attention_heads": nh,
        "num_key_value_heads": nk,
        "intermediate_size": I,
        "max_position_embeddings": 1024,
        "vocab_size": embed.shape[0],
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
    }

    hf = {}
    hf["model.embed_tokens.weight"] = embed.clone()
    hf["lm_head.weight"] = meg["output_layer.weight"].clone()
    hf["model.norm.weight"] = meg["decoder.final_layernorm.weight"].clone()

    proj_stacked = meg["decoder.layers.self_attention.linear_proj.weight"]      # [L, H, H]
    qkv_ln_stacked = meg["decoder.layers.self_attention.linear_qkv.layer_norm_weight"]  # [L, H]
    fc1_ln_stacked = meg["decoder.layers.mlp.linear_fc1.layer_norm_weight"]     # [L, H]
    fc2_stacked = meg["decoder.layers.mlp.linear_fc2.weight"]                    # [L, H, I]

    for l in range(L):
        hp = f"model.layers.{l}"

        q, k, v = convert_qkv_megatron_to_hf(qkv_stacked[l], nk, q_per_kv, head_dim, H)
        hf[f"{hp}.self_attn.q_proj.weight"] = q.contiguous()
        hf[f"{hp}.self_attn.k_proj.weight"] = k.contiguous()
        hf[f"{hp}.self_attn.v_proj.weight"] = v.contiguous()
        hf[f"{hp}.self_attn.o_proj.weight"] = proj_stacked[l].contiguous()

        hf[f"{hp}.input_layernorm.weight"] = qkv_ln_stacked[l].contiguous()
        hf[f"{hp}.post_attention_layernorm.weight"] = fc1_ln_stacked[l].contiguous()

        fc1 = fc1_stacked[l]  # [2I, H]
        hf[f"{hp}.mlp.w1.weight"] = fc1[:I, :].contiguous()
        hf[f"{hp}.mlp.w3.weight"] = fc1[I:, :].contiguous()
        hf[f"{hp}.mlp.w2.weight"] = fc2_stacked[l].contiguous()

    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    save_file({k: v.to(torch.bfloat16) for k, v in hf.items()},
              os.path.join(args.output, "model.safetensors"))

    for fn in ["__init__.py", "modeling_dense_llama.py", "configuration_dense_llama.py"]:
        src = os.path.join(args.model_files, fn)
        if os.path.exists(src):
            shutil.copy(src, args.output)
            print(f"  Copied {fn}")

    if args.tokenizer:
        for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "vocab.bpe", "encoder.json"]:
            src = os.path.join(args.tokenizer, fn)
            if os.path.exists(src):
                shutil.copy(src, args.output)

    print(f"Exported dense Llama checkpoint to: {args.output}")


if __name__ == "__main__":
    main()