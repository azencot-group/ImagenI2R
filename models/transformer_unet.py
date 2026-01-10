import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import silu
from models.TST import TSTransformerEncoder
from models.networks import UNetBlock, Linear, Conv2d, GroupNorm, PositionalEmbedding
from utils import persistence

@persistence.persistent_class
class TransformerToImageBridge(nn.Module):
    def __init__(self, t_dim, t_len, res, channels, ts_img):
        super().__init__()
        self.res = res
        self.channels = channels
        self.ts_img = ts_img
        self.proj = nn.Linear(t_dim, channels)

    def forward(self, x):
        # x: (batch, t_len, t_dim)
        batch = x.shape[0]
        # 1. Project tokens to target channels
        x = self.proj(x) # (batch, t_len, channels)
        # 2. Fold into image using DelayEmbedder logic
        img = self.ts_img.ts_to_img(x, pad=True) # (batch, channels, H, W)
        # 3. Resize to target resolution if needed
        if img.shape[-1] != self.res:
            img = torch.nn.functional.interpolate(img, size=(self.res, self.res), mode='bilinear', align_corners=False)
        return img

@persistence.persistent_class
class TransformerUNet(nn.Module):
    def __init__(self,
        img_resolution,                     # Target image resolution for decoder
        in_channels,                        # Number of TS features (input to transformer)
        out_channels,                       # Number of color channels at output of decoder
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels in CNN.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16, 8],      # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        # Transformer specific args
        feat_dim            = None,         # Should be in_channels
        max_len             = None,         # Should be seq_len
        d_model             = 128,
        n_heads             = 4,
        num_layers          = 3,
        dim_feedforward     = 512,
        ts_img              = None,         # DelayEmbedder instance
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Noise level injection for Transformer
        self.noise_proj = Linear(in_features=emb_channels, out_features=d_model, **init)

        # Transformer Encoder
        self.transformer_encoder = TSTransformerEncoder(
            feat_dim=feat_dim if feat_dim is not None else in_channels,
            max_len=max_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Encoder Skips emulation (to match DhariwalUNet decoder expectations)
        self.skips_info = []
        c_in_cnn = out_channels # Dummy start, we only care about output channels of blocks
        # Actually DhariwalUNet encoder starts with a conv then blocks
        
        # We need to calculate what the CNN encoder would have produced
        # Level 0
        res = img_resolution
        mult = channel_mult[0]
        # conv
        cout = model_channels * mult
        self.skips_info.append((res, cout))
        # blocks
        for idx in range(num_blocks):
            self.skips_info.append((res, cout))
        
        # Level 1+
        for level in range(1, len(channel_mult)):
            res = img_resolution >> level
            mult = channel_mult[level]
            # down
            self.skips_info.append((res, cout))
            # blocks
            for idx in range(num_blocks):
                cout = model_channels * mult
                self.skips_info.append((res, cout))

        # Bridges for skips
        self.bridges = nn.ModuleList()
        for res, channels in self.skips_info:
            self.bridges.append(TransformerToImageBridge(d_model, max_len, res, channels, ts_img))

        # Bridge for bottleneck
        final_res = img_resolution >> (len(channel_mult) - 1)
        final_channels = cout
        self.bottleneck_bridge = TransformerToImageBridge(d_model, max_len, final_res, final_channels, ts_img)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        curr_channels = final_channels
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=curr_channels, out_channels=curr_channels, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=curr_channels, out_channels=curr_channels, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=curr_channels, out_channels=curr_channels, up=True, **block_kwargs)
            
            for idx in range(num_blocks + 1):
                res_skip, channels_skip = self.skips_info.pop()
                cin = curr_channels + channels_skip
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                curr_channels = cout

        self.out_norm = GroupNorm(num_channels=curr_channels)
        self.out_conv = Conv2d(in_channels=curr_channels, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x_ts, noise_labels, class_labels, augment_labels=None, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones((x_ts.shape[0], x_ts.shape[1]), dtype=torch.bool, device=x_ts.device)

        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x_ts.shape[0], 1], device=x_ts.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Noise level injection for Transformer
        noise_emb = self.noise_proj(emb).unsqueeze(1) # (batch, 1, d_model)

        # Transformer Encoder
        t_out = self.transformer_encoder(x_ts, padding_masks, noise_emb=noise_emb) # (batch, t_len, d_model)

        # Create Skips
        skips = []
        for bridge in self.bridges:
            skips.append(bridge(t_out))

        # Bottleneck
        x = self.bottleneck_bridge(t_out)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, emb)
            else:
                x = block(x, emb)
        
        x = self.out_conv(silu(self.out_norm(x)))
        return x

