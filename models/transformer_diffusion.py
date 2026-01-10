import torch
import torch.nn as nn

from models.ema import LitEma
from models.our import TS2img_Karras
from models.transformer_unet import TransformerUNet
from utils import persistence

@persistence.persistent_class
class TransformerEDMPrecond(nn.Module):
    def __init__(self, model, img_to_ts, sigma_data=0.5):
        super().__init__()
        self.model = model
        self.img_to_ts = img_to_ts
        self.sigma_data = sigma_data
        self.sigma_min = 0
        self.sigma_max = float('inf')

    def forward(self, x_ts, sigma, class_labels=None, force_fp32=False, padding_masks=None):
        # x_ts: (B, T, C)
        x_ts = x_ts.to(torch.float32)
        # Ensure sigma is (B, 1, 1) for broadcasting with (B, T, C)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # TransformerUNet expects (batch, t_len, feat_dim)
        # c_noise should be flattened to (batch,)
        F_x_img = self.model((c_in * x_ts), c_noise.flatten(), class_labels=class_labels, padding_masks=padding_masks)
        
        # Convert back to TS
        F_x_ts = self.img_to_ts(F_x_img)
        
        # Result is in TS domain
        D_x = c_skip * x_ts + c_out * F_x_ts
        return D_x.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class TSTransformerDiffusion(TS2img_Karras):
    def __init__(self, args, device):
        super().__init__(args, device)
        
        # We replace the default net with our Transformer Hybrid
        d_model = getattr(args, 'd_model', 128)
        n_heads = getattr(args, 'n_heads', 4)
        
        # Ensure d_model is divisible by n_heads for MultiheadAttention
        if d_model % n_heads != 0:
            n_heads = 4 if d_model % 4 == 0 else 1
            print(f"Warning: Adjusted n_heads to {n_heads} to be compatible with d_model {d_model}")

        self.transformer_unet = TransformerUNet(
            img_resolution=args.img_resolution,
            in_channels=args.input_channels,
            out_channels=args.input_channels,
            model_channels=args.unet_channels,
            channel_mult=args.ch_mult,
            num_blocks=3, # Default matching DhariwalUNet
            feat_dim=args.input_channels,
            max_len=args.seq_len,
            d_model=d_model,
            num_layers=getattr(args, 'num_layers', 3),
            dim_feedforward=getattr(args, 'dim_feedforward', 512),
            n_heads=n_heads,
            dropout=getattr(args, 'dropout', 0.1),
            ts_img=self.ts_img
        ).to(device)

        # Wrap it with EDM preconditioning and TS-TS flow
        self.net = TransformerEDMPrecond(self.transformer_unet, self.img_to_ts, self.sigma_data)

        # Re-initialize EMA with the correct network structure
        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def loss_fn_regular(self, x_ts):
        # x_ts: (B, T, C) - clean time series
        # In run_regular.py, x_ts is provided as clean data
        
        rnd_normal = torch.randn([x_ts.shape[0], 1, 1], device=x_ts.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        noise = torch.randn_like(x_ts) * sigma
        x_ts_noisy = x_ts + noise
        
        # Prediction in TS domain
        x_ts_recon = self.net(x_ts_noisy, sigma)
        
        # Compute loss in TS domain
        loss = (weight * (x_ts_recon - x_ts).square()).mean()
        
        return loss, {'transformer_loss': loss.item()}

