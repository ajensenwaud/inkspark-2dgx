from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules.inkling2 import Inkling2Embedding, Inkling2Attention, Inkling2MoE, Inkling2Block, Inkling2Final, \
    Inkling2Head

# Inkling 2 (thinkingmachines/Inkling-2-Small, model_type inkling2), text decoder only. Reference: transformers
# models/inkling2 (PR #50682). Checkpoint namespace model.llm.*; routed experts stacked per layer as NVFP4
# (w13/w2 + e4m3 block scales, per-expert amax, outer_scale_denominator) or MXFP8 (ue8m0 scales), dequantised
# at load. The vision tower and the DSpark drafter (mtp.safetensors) are not part of this model: they ship
# beside the pack unquantised. Cache-free forward only (see modules/inkling2.py).


class Inkling2Config(Config):
    arch_string = "Inkling2ForConditionalGeneration"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": Inkling2Model}, config_key = "text_config", **kwargs)
        rd = self.read_cfg
        self.hidden_size = rd(int, "hidden_size", no_default)
        self.num_hidden_layers = rd(int, "num_hidden_layers", no_default)
        self.vocab_size = rd(int, "vocab_size", no_default)
        self.unpadded_vocab_size = rd(int, "unpadded_vocab_size", self.vocab_size)
        self.num_q_heads = rd(int, "num_attention_heads", no_default)
        self.num_kv_heads = rd(int, "num_key_value_heads", no_default)
        self.head_dim = rd(int, "head_dim", no_default)
        self.swa_heads = rd(int, "swa_num_attention_heads", self.num_q_heads)
        self.swa_kv_heads = rd(int, "swa_num_key_value_heads", self.num_kv_heads)
        self.swa_head_dim = rd(int, "swa_head_dim", self.head_dim)
        self.d_rel = rd(int, "d_rel", 16)
        self.rel_extent = rd(int, "rel_extent", 1024)
        self.sliding_window = rd(int, "sliding_window_size", 1024)
        self.log_floor = rd(int, "log_scaling_n_floor", 128000)
        self.log_alpha = rd(float, "log_scaling_alpha", 0.1)
        self.qk_clip = rd(float, "qk_l2_clip_threshold", 22.627416997969522)
        self.rel_clip = rd(float, "rel_l2_clip_threshold", 8.0)
        self.rms_norm_eps = rd(float, "rms_norm_eps", 1e-6)
        self.conv_kernel = rd(int, "sconv_kernel_size", 4)
        self.block_layers = rd(int, "n_layers_per_attn_res_block", 6)
        self.num_experts = rd(int, "n_routed_experts", no_default)
        self.num_experts_per_tok = rd(int, "num_experts_per_tok", no_default)
        self.moe_intermediate_size = rd(int, "intermediate_size", no_default)
        self.gate_cap = rd(float, "mlp_gate_softcap", 4.0)
        self.up_cap = rd(float, "mlp_up_softcap", 25.0)
        self.route_scale = rd(float, "route_scale", 8.0)
        self.logit_softcap = rd(float, "final_logit_softcapping", 30.0)
        self.max_position_embeddings = rd(int, "model_max_length", 1048576)
        self.assert_cfg(int, "n_shared_experts", 0, optional = True)
        self.assert_cfg(bool, "use_gate_bias", True, optional = True)
        self.assert_cfg(bool, "use_value_norm_no_gamma", True, optional = True)
        self.assert_cfg(bool, "attn_res_normalize_values", True, optional = True)
        self.assert_cfg(bool, "pre_norm_pre_affine", False, optional = True)
        self.assert_cfg(bool, "final_norm_pre_affine", False, optional = True)
        local = rd(list, "local_layer_ids", None)
        if local is None:
            local = [i for i in range(self.num_hidden_layers) if (i + 1) % self.block_layers]
        self.global_layers = [i for i in range(self.num_hidden_layers) if i not in set(local)]
        assert self.swa_head_dim == self.head_dim and self.swa_heads == self.num_q_heads \
            and self.swa_kv_heads == self.num_kv_heads, "Inkling2: sliding and global head shapes differ"

    @override
    def default_max_position_embeddings(self):
        return 1048576


class Inkling2Model(Model):
    config_class = Inkling2Config

    def __init__(self, config: Inkling2Config, **kwargs):
        super().__init__(config, **kwargs)
        c = config
        p = "model.llm"
        self.modules += [Inkling2Embedding(c, f"{p}.embed", f"{p}.embed_norm", c.rms_norm_eps)]
        self.first_block_idx = len(self.modules)
        for idx in range(c.num_hidden_layers):
            key = f"{p}.layers.{idx}"
            is_global = idx in c.global_layers
            attn = Inkling2Attention(
                c, f"{key}.attn", idx, is_global, c.hidden_size, c.num_q_heads, c.num_kv_heads, c.head_dim,
                c.d_rel, c.rel_extent if is_global else c.sliding_window, c.sliding_window, c.qk_clip,
                c.rel_clip, c.rms_norm_eps, c.log_floor, c.log_alpha,
            )
            moe = Inkling2MoE(
                c, f"{key}.mlp", c.hidden_size, c.moe_intermediate_size, c.num_experts, c.num_experts_per_tok,
                c.route_scale, c.gate_cap, c.up_cap,
            )
            self.modules += [Inkling2Block(
                c, key, idx, is_global, idx % c.block_layers == 0,
                f"{p}.attn_res.attn_queries.{idx - 1}" if idx > 0 else None,
                f"{p}.attn_res.mlp_queries.{idx}",
                attn, moe, c.rms_norm_eps,
            )]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [
            Inkling2Final(c, f"{p}.final", f"{p}.attn_res.output_query", f"{p}.norm", c.rms_norm_eps),
            Inkling2Head(c, f"{p}.unembed", c.hidden_size, c.vocab_size, c.logit_softcap),
        ]
        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True
        self.caps.update({"supports_tp": False})

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if params.get("cache") is not None or params.get("attn_mode", "flash_attn_nc") != "flash_attn_nc":
            raise NotImplementedError("Inkling2 in exllamav3 runs the cache-free forward only (attn_mode flash_attn_nc)")
        params["input_ids"] = input_ids
        params["ink_blocks"] = []
        params["ink_pos0"] = 0
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("use the checkpoint's chat_template.jinja")
