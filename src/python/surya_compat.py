"""
Compatibility shims for surya-ocr 0.17.x + transformers 5.5+.

surya was written against older transformers where:
  1. PretrainedConfig auto-set pad_token_id=None on every instance
  2. ROPE_INIT_FUNCTIONS contained a 'default' entry

Both were removed in transformers 5.x.  Apply these patches before
importing any surya module that touches models.
"""

import torch


def apply():
    # ── patch 1: pad_token_id not auto-set in transformers 5.5+ ──
    from surya.common.surya.decoder.config import SuryaDecoderConfig
    if not hasattr(SuryaDecoderConfig, "pad_token_id"):
        SuryaDecoderConfig.pad_token_id = None

    # ── patch 2: ROPE_INIT_FUNCTIONS lost its 'default' entry in 5.x ──
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope(config, device=None, seq_len=None):
            head_dim = getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
            inv_freq = 1.0 / (
                config.rope_theta
                ** (
                    torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float()
                    / head_dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope
