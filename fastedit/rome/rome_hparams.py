from dataclasses import dataclass
from typing import List

from ..utils.hparams import HyperParams


@dataclass
class ROMEHyperParams(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # MoE support
    is_moe: bool = False
    num_experts: int = 0
    # For MoE, we track representations at a different module than we edit
    # If None, uses rewrite_module_tmp for tracking (default for dense models)
    repr_module_tmp: str = None

    @classmethod
    def from_name(cls, name: str):
        data = dict(
            layers=[5],
            fact_token="subject_last",
            v_num_grad_steps=20,
            v_lr=1e-1,
            v_loss_layer=27,
            v_weight_decay=1e-3,
            clamp_norm_factor=4,
            kl_factor=0.0625,
            mom2_adjustment=False,
            rewrite_module_tmp="transformer.h.{}.mlp.fc_out",
            layer_module_tmp="transformer.h.{}",
            mlp_module_tmp="transformer.h.{}.mlp",
            attn_module_tmp="transformer.h.{}.attn",
            ln_f_module="transformer.ln_f",
            lm_head_module="lm_head",
            mom2_dataset="wikipedia",
            mom2_n_samples=100000,
            mom2_dtype="float16",
            is_moe=False,
            num_experts=0,
            repr_module_tmp=None
        )

        if name == "gpt-j-6b":
            pass
        elif name == "llama-7b":
            r"""
            Supports: LLaMA-7B, LLaMA-2-7B, Baichuan-7B, InternLM-7B...
            """
            data.update(dict(
                v_loss_layer=31,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name == "llama-13b":
            r"""
            Supports LLaMA-13B, LLaMA-2-13B, Baichuan-13B...
            """
            data.update(dict(
                layers=[10],
                v_loss_layer=39,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name == "falcon-7b":
            data.update(dict(
                v_loss_layer=31,
                rewrite_module_tmp="transformer.h.{}.mlp.dense_4h_to_h",
                attn_module_tmp="transformer.h.{}.self_attention"
            ))
        elif name == "bloom-7b1":
            data.update(dict(
                v_lr=2e-1,
                v_loss_layer=29,
                rewrite_module_tmp="transformer.h.{}.mlp.dense_4h_to_h",
                attn_module_tmp="transformer.h.{}.self_attention"
            ))
        elif name in ("qwen3-0.6b", "qwen3-1.7b", "qwen3.5-0.8b", "qwen3.5-2b"):
            r"""
            Supports: Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-0.8B, Qwen3.5-2B (28 layers)
            """
            data.update(dict(
                v_loss_layer=27,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name in ("qwen3-4b", "qwen3-8b", "qwen3.5-4b", "qwen3.5-9b"):
            r"""
            Supports: Qwen3-4B, Qwen3-8B, Qwen3.5-4B, Qwen3.5-9B (36 layers)
            """
            data.update(dict(
                layers=[7],
                v_loss_layer=35,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name in ("qwen3-14b", "qwen3.5-27b"):
            r"""
            Supports: Qwen3-14B, Qwen3.5-27B (40 layers)
            """
            data.update(dict(
                layers=[10],
                v_loss_layer=39,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name in ("qwen3-32b",):
            r"""
            Supports: Qwen3-32B (64 layers)
            """
            data.update(dict(
                layers=[16],
                v_loss_layer=63,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name in ("qwen3-30b-a3b",):
            r"""
            Supports: Qwen3-30B-A3B MoE (48 layers, 128 experts, 8 activated)
            """
            data.update(dict(
                layers=[12],
                v_loss_layer=47,
                rewrite_module_tmp="model.layers.{}.mlp.experts.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm",
                is_moe=True,
                num_experts=128,
                repr_module_tmp="model.layers.{}.mlp.experts"
            ))
        elif name in ("qwen3-235b-a22b",):
            r"""
            Supports: Qwen3-235B-A22B MoE (94 layers, 128 experts, 8 activated)
            """
            data.update(dict(
                layers=[24],
                v_loss_layer=93,
                rewrite_module_tmp="model.layers.{}.mlp.experts.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm",
                is_moe=True,
                num_experts=128,
                repr_module_tmp="model.layers.{}.mlp.experts"
            ))
        elif name in ("qwen2-7b", "qwen2.5-7b"):
            r"""
            Supports: Qwen2-7B, Qwen2.5-7B (28 layers)
            """
            data.update(dict(
                v_loss_layer=27,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name in ("qwen2-14b", "qwen2.5-14b"):
            r"""
            Supports: Qwen2-14B, Qwen2.5-14B (40 layers)
            """
            data.update(dict(
                layers=[10],
                v_loss_layer=39,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name in ("qwen2-32b", "qwen2.5-32b"):
            r"""
            Supports: Qwen2-32B, Qwen2.5-32B (64 layers)
            """
            data.update(dict(
                layers=[16],
                v_loss_layer=63,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name in ("qwen2-72b", "qwen2.5-72b"):
            r"""
            Supports: Qwen2-72B, Qwen2.5-72B (80 layers)
            """
            data.update(dict(
                layers=[20],
                v_loss_layer=79,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name in ("qwen3.5-35b-a3b",):
            r"""
            Supports: Qwen3.5-35B-A3B MoE (40 layers, 256 experts, 8 activated)
            """
            data.update(dict(
                layers=[10],
                v_loss_layer=39,
                rewrite_module_tmp="model.layers.{}.mlp.experts.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm",
                is_moe=True,
                num_experts=256,
                repr_module_tmp="model.layers.{}.mlp.experts"
            ))
        elif name in ("qwen3.5-122b-a10b",):
            r"""
            Supports: Qwen3.5-122B-A10B MoE (48 layers, 256 experts, 8 activated)
            """
            data.update(dict(
                layers=[12],
                v_loss_layer=47,
                rewrite_module_tmp="model.layers.{}.mlp.experts.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm",
                is_moe=True,
                num_experts=256,
                repr_module_tmp="model.layers.{}.mlp.experts"
            ))
        else:
            raise NotImplementedError

        return cls(**data)
