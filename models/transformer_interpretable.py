import torch
import torch.nn as nn
import os
import sys

from models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS

class TSTransformerInterpretable(Diffusion_TS):
    def __init__(self, args, device):
        # Extract parameters from args or use defaults from Diffusion-TS config
        seq_length = args.seq_len
        feature_size = args.input_channels
        
        # Use reasonable defaults or args if available
        n_layer_enc = getattr(args, 'n_layer_enc', 1)
        n_layer_dec = getattr(args, 'n_layer_dec', 2)
        d_model = getattr(args, 'd_model', 64)
        timesteps = getattr(args, 'diffusion_steps', 200) # DDPM timesteps
        sampling_timesteps = getattr(args, 'sampling_timesteps', timesteps)
        n_heads = getattr(args, 'n_heads', 4)

        # Ensure d_model is divisible by n_heads for MultiheadAttention
        if d_model % n_heads != 0:
            n_heads = 4 if d_model % 4 == 0 else 1
            print(f"Warning: Adjusted n_heads to {n_heads} to be compatible with d_model {d_model}")
        
        super().__init__(
            seq_length=seq_length,
            feature_size=feature_size,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            d_model=d_model,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=n_heads,
            mlp_hidden_times=4,
            attn_pd=0.1,
            resid_pd=0.1,
            kernel_size=5,
            padding_size=2
        )
        self.to(device)
        self.device = device

    def loss_fn_regular(self, x_ts):
        # x_ts: (B, T, C)
        # Diffusion_TS.forward returns the average loss over the batch
        loss = self.forward(x_ts)
        return loss, {'interpretable_loss': loss.item()}

    def sample_ts(self, batch_size):
        # generate_mts returns (B, T, C)
        return self.generate_mts(batch_size=batch_size)

