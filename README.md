# CoRM: Contrastive Routing Mechanism

Official implementation of ["Beyond Magnitude: Contrastive Routing for Modular Mixture-of-Experts"](https://arxiv.org/abs/2609.01100).

> [!WARNING]
> This codebase is currently under development and not yet ready for general use.

## Abstract

> In current Mixture-of-Experts architectures, routing is performed based on representations dominated by structure shared across all tokens, limiting expert specialization.  We show that contrasting each token against an 
Exponential Moving Average of the layer's hidden states, rather than routing on absolute magnitude, concentrates the routing signal onto a low-dimensional, highly separable subspace. Building on this, we propose the Contrastive Routing Mechanism (CoRM), which scores each expert by the gap between its affinity for the incoming token and its affinity for this shared reference state, interpreted through a distinct per-expert projection. The resulting experts have routing boundaries that align with linguistic structure significantly more than the Top-$k$ baseline. Our experiments show that CoRM leads to a $+1.15\%$ (Top-1) and $+1.65\%$ (Top-2) mean absolute improvement on nine zero-shot reasoning benchmarks, at the minimal cost of $2.9\%$ added parameters and $2.6\%$ added FLOPs per token.

<p align="center">
  <img src="assets/CoRM_dark.png" width="700"/>
  <!-- <img src="assets/CoRM.png" width="700"/> -->
</p>  

This codebase builds on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [ReMoE](https://github.com/thu-ml/ReMoE) and supports Data, Tensor, Pipeline, and Expert Parallelism.

## Installation

Same as Megatron-LM. See the [NGC Pytorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) or install manually:

```bash
conda create -n corm python=3.11
conda activate corm
pip install torch torchvision torchaudio numpy
pip install packaging ninja
pip install flash-attn --no-build-isolation

# Apex
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# TransformerEngine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```



## Usage


### Data

Training uses The Pile dataset, tokenized with the GPT-2 tokenizer.

### Reproducing from Scratch

1. **Download** the Pile dataset from [Hugging Face](https://huggingface.co/datasets/monology/pile-uncopyrighted).
2. **Preprocess** with:
   ```bash
   bash data_perprocessing.sh
   ```
   Some minor modifications are required, such as the path to vocabulary files (`gpt2-vocab.json`, `gpt2-merges.txt`).
3. **Training** Run the following script for pretraining CoRM models:
   ```bash
   bash scripts/train_llama_182m_corm_top1.sh
   ```



## Checkpoints

Checkpoints are available on [Hugging Face](https://huggingface.co/ilsp/CoRM)

See the [paper](https://arxiv.org/abs/2609.01100) for full results, ablations, and analysis.

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@misc{xiros2026magnitudecontrastiveroutingmodular,
      title={Beyond Magnitude: Contrastive Routing for Modular Mixture-of-Experts}, 
      author={Nikolaos Xiros and Dimitrios Damianos and Maria-Eleni Zoumpoulidi and Leon Voukoutis and Vassilis Katsouros and Georgios Paraskevopoulos},
      year={2026},
      eprint={2609.01100},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.01100}, 
}

```

## License

This project is licensed under the Apache 2.0 License — see [LICENSE](LICENSE) for details.


