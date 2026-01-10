import torch
import torch.nn as nn
from models.our import TS2img_Karras
from models.interpretable_diffusion.transformer_irregular import Transformer
from utils import persistence
from models.ema import LitEma

@persistence.persistent_class
class InterpretableTransformerEDMPrecondIrregular(nn.Module):
    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = 0
        self.sigma_max = float('inf')

    def forward(self, x_ts, sigma, class_labels=None, force_fp32=False):
        # x_ts: (B, T, C) - may contain NaNs
        x_ts = x_ts.to(torch.float32)
        
        # 1. Compute padding mask (B, T) - True where NOT all features are NaN
        # We assume if a time point is missing, all features are NaN
        padding_masks = ~torch.isnan(x_ts).any(dim=-1)
        
        # 2. Replace NaNs with 0 for backbone processing (Conv/FFT)
        x_ts_clean = x_ts.clone()
        x_ts_clean[torch.isnan(x_ts_clean)] = 0
        
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Forward through the full Interpretable Transformer with mask
        trend, season_error = self.model((c_in * x_ts_clean), c_noise.flatten(), padding_masks=padding_masks)
        F_x_ts = trend + season_error
        
        # Result is in TS domain
        D_x = c_skip * x_ts_clean + c_out * F_x_ts
        
        # Restore NaNs in the output to be consistent with input for evaluation/loss
        # (Though loss_fn will handle it)
        D_x[~padding_masks] = float('nan')
        
        return D_x.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class TSTransformerInterpretableEDMIrregular(TS2img_Karras):
    def __init__(self, args, device):
        super().__init__(args, device)
        
        # Initialize the full Interpretable Transformer backbone
        d_model = getattr(args, 'd_model', 128)
        n_heads = getattr(args, 'n_heads', 4)

        # Ensure d_model is divisible by n_heads for MultiheadAttention
        if d_model % n_heads != 0:
            n_heads = 4 if d_model % 4 == 0 else 1
            print(f"Warning: Adjusted n_heads to {n_heads} to be compatible with d_model {d_model}")

        self.interpretable_transformer = Transformer(
            n_feat=args.input_channels,
            n_channel=args.seq_len,
            n_layer_enc=getattr(args, 'n_layer_enc', 3),
            n_layer_dec=getattr(args, 'n_layer_dec', 6),
            n_embd=d_model,
            n_heads=n_heads,
            attn_pdrop=getattr(args, 'dropout', 0.1),
            resid_pdrop=getattr(args, 'dropout', 0.1),
            mlp_hidden_times=4,
            max_len=args.seq_len,
            conv_params=[getattr(args, 'kernel_size', 5), getattr(args, 'padding_size', 2)]
        ).to(device)

        # Wrap it with EDM preconditioning
        self.net = InterpretableTransformerEDMPrecondIrregular(self.interpretable_transformer, self.sigma_data)

        # Re-initialize EMA with the correct network structure
        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def loss_fn_irregular(self, x_ts):
        # x_ts: (B, T, C) - clean time series WITH NaNs
        rnd_normal = torch.randn([x_ts.shape[0], 1, 1], device=x_ts.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        # 1. Mask for known values (B, T, C)
        mask = ~torch.isnan(x_ts)
        
        # 2. Add noise only to known values
        noise = torch.randn_like(x_ts) * sigma
        x_ts_noisy = x_ts.clone()
        x_ts_noisy[mask] = x_ts[mask] + noise[mask]
        
        # 3. Prediction (D_x will have NaNs where input had NaNs)
        x_ts_recon = self.net(x_ts_noisy, sigma)
        
        # 4. Masked Loss calculation
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        # weight: (B, 1, 1), x_ts: (B, T, C)
        sq_error = (x_ts_recon - x_ts).square()
        
        # Only take mean over non-NaN elements
        loss = (weight * sq_error[mask]).mean()
        
        return loss, {'interpretable_edm_irregular_loss': loss.item()}

