from transformers import PretrainedConfig


class CoRMConfig(PretrainedConfig):
    """Config for CoRM (Contrastive Routing Mixture-of-Experts).

    The only supported `router_type` is "corm" — a contrastive attention
    router that scores each expert by the gap between its query's attention
    to the real token and to an EMA-tracked null (background) state.
    """

    model_type = "corm"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=3072,
        num_local_experts=8,
        num_experts_per_tok=2,
        # Dynamic top-p routing was not used for the released models and is
        # not implemented in modeling_corm.py (routing is always fixed top-k).
        use_dynamic_routing=False,
        top_p=1.0,
        expert_attn_size=64,
        max_position_embeddings=1024,
        vocab_size=51200,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        routing_temperature_init=1.0,

        # Router variant selection — only "corm" is supported.
        router_type="corm",

        # CoRM router: EMA momentum for the background null state.
        router_background_momentum=0.01,

        # CoRM router: L2-normalize K, Q_real and Q_null before the dot product
        # (paper setting; matches Megatron's moe_router_use_l2_normalize).
        moe_router_use_l2_normalize=True,

        # Weight-schedule params (inert for the corm router, which has no
        # shared expert; kept for backward compatibility with released
        # config.json files that still carry these keys).
        moe_weight_schedule_warmup_stage1=1000,
        moe_weight_schedule_warmup_stage2=5000,
        moe_weight_schedule_target_w_shared=1.0,
        moe_weight_schedule_transition_type="cosine",

        # Shared/routed expert output-combination weights (see note above).
        shared_expert_weight=0.5,
        routed_expert_weight=0.5,
        learn_output_weights=False,  # If True, learn shared_expert_weight/routed_expert_weight
        fixed_output_weights=True,   # If True and not learn_output_weights, use fixed weights

        # Whether this model has a separate shared expert MLP
        # (the corm router does NOT have one in Megatron).
        has_shared_expert=None,  # Auto-inferred from router_type if None

        torch_dtype="bfloat16",
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_dynamic_routing = use_dynamic_routing
        self.top_p = top_p
        self.expert_attn_size = expert_attn_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.routing_temperature_init = routing_temperature_init

        # Shared expert: auto-infer if not explicitly set.
        # (the corm router has no shared expert unless explicitly configured otherwise.)
        if has_shared_expert is None:
            self.has_shared_expert = False
        else:
            self.has_shared_expert = has_shared_expert

        # Router variant configuration
        self.router_type = router_type

        self.router_background_momentum = router_background_momentum
        self.moe_router_use_l2_normalize = moe_router_use_l2_normalize

        # Scheduling params
        self.moe_weight_schedule_warmup_stage1 = moe_weight_schedule_warmup_stage1
        self.moe_weight_schedule_warmup_stage2 = moe_weight_schedule_warmup_stage2
        self.moe_weight_schedule_target_w_shared = moe_weight_schedule_target_w_shared
        self.moe_weight_schedule_transition_type = moe_weight_schedule_transition_type

        # Shared expert weights (for output combination)
        self.shared_expert_weight = shared_expert_weight
        self.routed_expert_weight = routed_expert_weight
        self.learn_output_weights = learn_output_weights
        self.fixed_output_weights = fixed_output_weights

        self.torch_dtype = torch_dtype

        # Must be passed to super(): PretrainedConfig.__init__ otherwise resets it
        # to True, which ties lm_head to embed_tokens and discards the trained lm_head.
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
