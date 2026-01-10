import torch
import torch.nn as nn
import torch.nn.functional as F
from models.our import TS2img_Karras
from models.diffusion_ts import TrendBlock, FourierLayer

class STLSynthesisModel(TS2img_Karras):
    """
    STLSynthesisModel combines image-based residual diffusion with 
    model-driven trend and seasonality synthesis.
    """
    def __init__(self, args, device):
        super().__init__(args, device)
        
        # Backbone for extracting features from noisy time series for T and S synthesis
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(args.input_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Trend Synthesis Head (polynomial regressor like Diffusion-TS)
        self.trend_head = TrendBlock(
            in_dim=128, 
            out_dim=args.seq_len, 
            in_feat=args.seq_len, 
            out_feat=args.input_channels, 
            act=nn.GELU()
        )
        
        # Seasonality Synthesis Head (Fourier synthetic layer like Diffusion-TS)
        self.season_head = FourierLayer(d_model=128)
        self.season_proj = nn.Linear(128, args.input_channels)

    def synthesize_components(self, x_ts_noisy):
        """
        Synthesize trend and seasonal components from the noisy input.
        x_ts_noisy: (B, T, C)
        """
        # (B, T, C) -> (B, C, T) -> (B, 128, T)
        features = self.feature_extractor(x_ts_noisy.transpose(1, 2))
        
        # Trend: (B, 128, T) -> (B, T, C)
        trend = self.trend_head(features)
        
        # Seasonality: (B, T, 128) -> (B, T, C)
        season_raw = self.season_head(features.transpose(1, 2))
        season = self.season_proj(season_raw)
        
        return trend, season

    def forward(self, x_ts_noisy, sigma, labels=None, augment_labels=None):
        """
        Denoising forward pass compatible with DiffusionProcess.
        Predicts the full reconstructed x0 by combining synthesized components.
        """
        # Ensure sigma has correct shape for U-Net
        if sigma.ndim == 1:
            sigma = sigma.view(-1, 1, 1, 1)
            
        # 1. Synthesize Trend and Seasonality from the noisy input
        trend, season = self.synthesize_components(x_ts_noisy)
        
        # 2. Process Residual using image-based diffusion (as in run_regular_stl.py)
        # Convert noisy full signal to image representation
        x_img_noisy = self.ts_to_img(x_ts_noisy)
        
        # U-Net predicts the denoised residual image
        # Note: In our setup, the U-Net is trained to match the STL residual
        res_img_denoised = self.net(x_img_noisy, sigma, labels, augment_labels=augment_labels)
        
        # Convert back to time series
        res_ts_denoised = self.img_to_ts(res_img_denoised)
        
        # 3. Combine synthesized components with denoised residual
        x0_hat = trend + season + res_ts_denoised
        
        return x0_hat

    def round_sigma(self, sigma):
        return sigma


