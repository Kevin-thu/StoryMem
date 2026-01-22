# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan m2v A14B ------------------------#

m2v_A14B = EasyDict(__name__='Config: Wan M2V A14B')
m2v_A14B.update(wan_shared_cfg)

m2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
m2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
m2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
m2v_A14B.vae_stride = (4, 8, 8)

# transformer
m2v_A14B.patch_size = (1, 2, 2)
m2v_A14B.dim = 5120
m2v_A14B.ffn_dim = 13824
m2v_A14B.freq_dim = 256
m2v_A14B.num_heads = 40
m2v_A14B.num_layers = 40
m2v_A14B.window_size = (-1, -1)
m2v_A14B.qk_norm = True
m2v_A14B.cross_attn_norm = True
m2v_A14B.eps = 1e-6
m2v_A14B.low_noise_checkpoint = 'low_noise_model'
m2v_A14B.high_noise_checkpoint = 'high_noise_model'

#lora
m2v_A14B.low_noise_lora = {
    'r': 128,
    'lora_alpha': 128,
    'lora_dropout': 0.05,
    'bias': 'none',
    'target_modules': ['self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o','cross_attn.q', 'cross_attn.k', 'cross_attn.v', 'cross_attn.o', 'ffn.0', 'ffn.2'],
    'use_rslora': True,
    'enabled': True
}
m2v_A14B.high_noise_lora = {
    'r': 128,
    'lora_alpha': 128,
    'lora_dropout': 0.05,
    'bias': 'none',
    'target_modules': ['self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o','cross_attn.q', 'cross_attn.k', 'cross_attn.v', 'cross_attn.o', 'ffn.0', 'ffn.2'],
    'use_rslora': True,
    'enabled': True
}

# inference
m2v_A14B.sample_shift = 4.0
m2v_A14B.sample_steps = 40
m2v_A14B.boundary = 0.900
m2v_A14B.sample_guide_scale = (3.5, 3.5)  # low noise, high noise
