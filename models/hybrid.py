import torch
import torch.nn as nn
import torch.nn.functional as F
from models.our import TS2img_Karras
from models.diffusion_ts import TrendBlock, FourierLayer
from contextlib import contextmanager

class HybridDiffusionModel(TS2img_Karras):
    def __init__(self, args, device):
        super().__init__(args, device)
        
        # --- Synthesis Heads for Trend and Seasonality ---
        # Feature extraction for T and S synthesis
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(args.input_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Trend Synthesis Head
        self.trend_head = TrendBlock(
            in_dim=128, 
            out_dim=args.seq_len, 
            in_feat=args.seq_len, 
            out_feat=args.input_channels, 
            act=nn.GELU()
        )
        
        # Seasonality Synthesis Head
        self.season_head = FourierLayer(d_model=128)
        self.season_proj = nn.Linear(128, args.input_channels)

    def synthesize_components(self, x_ts):
        """
        Synthesize Trend and Seasonality from time series input.
        x_ts: (B, T, C)
        """
        # features: (B, 128, T)
        features = self.feature_encoder(x_ts.transpose(1, 2))
        
        # trend: (B, T, C)
        trend = self.trend_head(features)
        
        # seasonality: (B, T, C)
        season_raw = self.season_head(features.transpose(1, 2))
        season = self.season_proj(season_raw)
        
        return trend, season

    def forward(self, x_ts_noisy, sigma, labels=None, augment_labels=None):
        """
        Denoising forward pass for DiffusionProcess.
        x_ts_noisy: (B, T, C)
        sigma: noise levels
        Returns: reconstructed clean signal estimate (B, T, C)
        """
        # 1. Synthesize Trend and Seasonality from the noisy input
        trend, season = self.synthesize_components(x_ts_noisy)
        
        # 2. Process Residual using image-based diffusion
        # Convert noisy full signal to image representation
        x_img_noisy = self.ts_to_img(x_ts_noisy)
        
        # U-Net predicts the denoised residual image
        res_img_denoised = self.net(x_img_noisy, sigma, labels, augment_labels=augment_labels)
        
        # Convert back to time series
        res_ts_denoised = self.img_to_ts(res_img_denoised)
        
        # 3. Combine components
        x0_hat = trend + season + res_ts_denoised
        
        return x0_hat

    def loss_fn_hybrid(self, x_ts_full, trends_gt, seasonals_gt, residuals_gt):
        """
        Integrated loss:
        1. Karras loss on residuals (matching STL residuals)
        2. MSE loss on synthesized trend
        3. MSE loss on synthesized seasonality
        """
        # Sample noise level
        rnd_normal = torch.randn([x_ts_full.shape[0], 1, 1], device=x_ts_full.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        # 1. Residual Training (Karras style)
        # Convert STL residuals to images
        residuals_img_gt = self.ts_to_img(residuals_gt)
        
        # Add noise only to the residual part for the "diffusion learning" part?
        # No, let's noise the FULL signal to simulate real sampling conditions
        noise_full = torch.randn_like(x_ts_full) * sigma
        x_ts_noisy = x_ts_full + noise_full
        
        # Synthesize components from noisy input
        trend_pred, season_pred = self.synthesize_components(x_ts_noisy)
        
        # Get residual prediction from U-Net (via image conversion)
        x_img_noisy = self.ts_to_img(x_ts_noisy)
        res_img_pred = self.net(x_img_noisy, sigma)
        
        # Residual loss: compare predicted residual image with GT residual image
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # res_img_pred should match the clean residual image
        res_loss = (weight.unsqueeze(-1) * (res_img_pred - residuals_img_gt).square()).mean()
        
        # 2. Trend & Seasonality matching
        trend_loss = F.mse_loss(trend_pred, trends_gt)
        season_loss = F.mse_loss(season_pred, seasonals_gt)
        
        total_loss = res_loss + 0.5 * trend_loss + 0.5 * season_loss
        
        to_log = {
            'hybrid_loss': total_loss.item(),
            'res_loss': res_loss.item(),
            'trend_loss': trend_loss.item(),
            'season_loss': season_loss.item()
        }
        
        return total_loss, to_log

    # For compatibility with DiffusionProcess
    def round_sigma(self, sigma):
        return sigma
