import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from utils.utils_stl import STLComponentStorage, auto_detect_period

class LinearTrendModel:
    """
    Fits and generates a linear trend: y = ax + b
    """
    def fit(self, trend_comp):
        """
        trend_comp: (T,) or (T, C)
        Returns: params (2*C,) where even indices are 'a' and odd are 'b'
        """
        T = trend_comp.shape[0]
        C = trend_comp.shape[1] if len(trend_comp.shape) > 1 else 1
        
        if len(trend_comp.shape) == 1:
            trend_comp = trend_comp.unsqueeze(1)
            
        x = torch.linspace(0, 1, T, device=trend_comp.device).unsqueeze(1) # (T, 1)
        X = torch.cat([x, torch.ones_like(x)], dim=1) # (T, 2)
        
        # Solve (X^T X)^-1 X^T Y
        # X: (T, 2), Y: (T, C) -> params: (2, C)
        params = torch.linalg.lstsq(X, trend_comp).solution # (2, C)
        return params.T.reshape(-1) # (2*C,)

    def generate(self, T, params, device='cpu'):
        """
        params: (2*C,)
        Returns: trend (T, C)
        """
        C = params.shape[0] // 2
        params = params.reshape(C, 2)
        x = torch.linspace(0, 1, T, device=device).unsqueeze(1) # (T, 1)
        X = torch.cat([x, torch.ones_like(x)], dim=1) # (T, 2)
        trend = X @ params.T # (T, C)
        return trend

class FourierSeasonalityModel:
    """
    Fits and generates seasonality using Fourier series.
    S(t) = sum_{k=1}^K (a_k cos(2pi*k*t/P) + b_k sin(2pi*k*t/P))
    """
    def __init__(self, period, num_harmonics=3):
        self.period = period
        self.num_harmonics = num_harmonics

    def _get_basis(self, T, device):
        t = torch.arange(T, device=device).float()
        basis = []
        for k in range(1, self.num_harmonics + 1):
            basis.append(torch.cos(2 * np.pi * k * t / self.period).unsqueeze(1))
            basis.append(torch.sin(2 * np.pi * k * t / self.period).unsqueeze(1))
        return torch.cat(basis, dim=1) # (T, 2*K)

    def fit(self, season_comp):
        """
        season_comp: (T, C)
        Returns: params (2*K*C,)
        """
        T, C = season_comp.shape
        X = self._get_basis(T, season_comp.device) # (T, 2*K)
        
        # Solve (X^T X)^-1 X^T Y
        params = torch.linalg.lstsq(X, season_comp).solution # (2*K, C)
        return params.T.reshape(-1) # (2*K*C,)

    def generate(self, T, params, device='cpu'):
        """
        params: (2*K*C,)
        """
        C = params.shape[0] // (2 * self.num_harmonics)
        params = params.reshape(C, 2 * self.num_harmonics)
        X = self._get_basis(T, device) # (T, 2*K)
        season = X @ params.T # (T, C)
        return season

class LearnedSTLComponentStorage(STLComponentStorage):
    """
    Extends STLComponentStorage to fit parametric models and sample from learned distributions.
    """
    def __init__(self, device='cpu', num_harmonics=3):
        super().__init__(device=device)
        self.num_harmonics = num_harmonics
        self.trend_model = LinearTrendModel()
        self.season_model = None # Initialized in finalize
        
        self.trend_dist = None
        self.season_dist = None
        self.seq_len = None
        self.n_channels = None

    def finalize(self):
        """
        Fits models to all stored components and estimates parameter distributions.
        """
        if len(self.trends) == 0:
            raise RuntimeError("No components stored")
            
        # Concatenate batches
        trends = torch.cat(self.trends, dim=0) # (N, T, C)
        seasonals = torch.cat(self.seasonals, dim=0) # (N, T, C)
        
        N, T, C = trends.shape
        self.seq_len = T
        self.n_channels = C
        
        period = auto_detect_period(T)
        self.season_model = FourierSeasonalityModel(period, self.num_harmonics)
        
        all_trend_params = []
        all_season_params = []
        
        print(f"Fitting parametric models to {N} samples...")
        for i in range(N):
            t_params = self.trend_model.fit(trends[i].to(self.device))
            s_params = self.season_model.fit(seasonals[i].to(self.device))
            all_trend_params.append(t_params.cpu())
            all_season_params.append(s_params.cpu())
            
        all_trend_params = torch.stack(all_trend_params) # (N, 2*C)
        all_season_params = torch.stack(all_season_params) # (N, 2*K*C)
        
        # Estimate distributions (Multivariate Normal)
        self.trend_dist = self._estimate_dist(all_trend_params)
        self.season_dist = self._estimate_dist(all_season_params)
        
        self.is_finalized = True
        print("Finalized learned STL storage.")

    def _estimate_dist(self, params):
        """
        Estimates a MultivariateNormal distribution from parameters.
        Adds small epsilon to diagonal for stability.
        """
        mean = params.mean(dim=0)
        # Covariance with shrinkage for stability
        cov = torch.from_numpy(np.cov(params.numpy(), rowvar=False)).float()
        eps = 1e-4 * torch.eye(cov.shape[0])
        return MultivariateNormal(mean, cov + eps)

    def sample_learned(self, batch_size):
        """
        Samples trend and seasonal components from learned distributions.
        """
        if not self.is_finalized:
            raise RuntimeError("Storage must be finalized before sampling")
            
        # Sample parameters
        t_params = self.trend_dist.sample((batch_size,)).to(self.device) # (B, 2*C)
        s_params = self.season_dist.sample((batch_size,)).to(self.device) # (B, 2*K*C)
        
        sampled_trends = []
        sampled_seasonals = []
        
        for i in range(batch_size):
            sampled_trends.append(self.trend_model.generate(self.seq_len, t_params[i], self.device))
            sampled_seasonals.append(self.season_model.generate(self.seq_len, s_params[i], self.device))
            
        return torch.stack(sampled_trends), torch.stack(sampled_seasonals)

