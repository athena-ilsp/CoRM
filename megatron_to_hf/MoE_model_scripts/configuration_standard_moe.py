from transformers import PretrainedConfig

class StandardMoEConfig(PretrainedConfig):
    model_type = "standard_moe"

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=3072,
        num_local_experts=8,
        num_experts_per_tok=1,
        shared_expert_intermediate_size=0,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        xmoe_expert_dim=4,
        **kwargs,
    ):
        self.xmoe_expert_dim = xmoe_expert_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)