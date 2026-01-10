import torch
import torch.nn as nn
from models.our import TS2img_Karras
from models.interpretable_diffusion.transformer import Transformer
from utils import persistence
from models.ema import LitEma

@persistence.persistent_class
class InterpretableTransformerEDMPrecond(nn.Module):
    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = 0
        self.sigma_max = float('inf')

    def forward(self, x_ts, sigma, class_labels=None, force_fp32=False, padding_masks=None):
        # x_ts: (B, T, C)
        x_ts = x_ts.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Forward through the full Interpretable Transformer
        # Transformer returns (trend, season_error)
        # All inputs to self.model must be float32
        trend, season_error = self.model((c_in * x_ts), c_noise.flatten(), padding_masks=padding_masks)
        F_x_ts = trend + season_error
        
        # Result is in TS domain
        D_x = c_skip * x_ts + c_out * F_x_ts
        return D_x.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class TSTransformerInterpretableEDM(TS2img_Karras):
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
        self.net = InterpretableTransformerEDMPrecond(self.interpretable_transformer, self.sigma_data)

        # Re-initialize EMA with the correct network structure
        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def loss_fn_regular(self, x_ts):
        # x_ts: (B, T, C) - clean time series
        rnd_normal = torch.randn([x_ts.shape[0], 1, 1], device=x_ts.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        noise = torch.randn_like(x_ts) * sigma
        x_ts_noisy = x_ts + noise
        
        # Prediction in TS domain
        x_ts_recon = self.net(x_ts_noisy, sigma)
        
        # Compute loss in TS domain
        loss = (weight * (x_ts_recon - x_ts).square()).mean()
        
        return loss, {'interpretable_edm_loss': loss.item()}

