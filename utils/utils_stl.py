"""
STL Decomposition Utility Functions
Provides STL decomposition with auto-period detection for time series data
"""
import numpy as np
import torch
from statsmodels.tsa.seasonal import STL


def auto_detect_period(seq_len):
    """
    Auto-detect the period for STL decomposition based on sequence length.
    
    Args:
        seq_len: Length of the time series sequence
        
    Returns:
        period: Detected period for STL decomposition (must be odd)
    """
    # Heuristic: use a fraction of sequence length as period
    # Common periods: 12 (monthly), 24 (hourly), 7 (weekly)
    if seq_len >= 48:
        period = 24
    elif seq_len >= 24:
        period = 12
    elif seq_len >= 14:
        period = 7
    else:
        # For short sequences, use a smaller period
        period = max(3, seq_len // 4)
    
    # STL requires period to be odd and >= 3
    if period % 2 == 0:
        period += 1
    period = max(3, period)
    
    # Period must be less than seq_len
    if period >= seq_len:
        period = max(3, (seq_len // 2) - 1)
        if period % 2 == 0:
            period -= 1
    
    return period


def stl_decompose_single(ts, period=None, robust=True):
    """
    Apply STL decomposition to a single univariate time series.
    
    Args:
        ts: 1D numpy array or torch tensor of shape [seq_len]
        period: Period for STL decomposition. If None, auto-detected.
        robust: Whether to use robust STL (handles outliers better)
        
    Returns:
        trend: Trend component [seq_len]
        seasonal: Seasonal component [seq_len]
        residual: Residual component [seq_len]
    """
    # Convert to numpy if torch tensor
    is_torch = isinstance(ts, torch.Tensor)
    if is_torch:
        device = ts.device
        ts_np = ts.detach().cpu().numpy()
    else:
        ts_np = ts
    
    seq_len = len(ts_np)
    
    # Auto-detect period if not provided
    if period is None:
        period = auto_detect_period(seq_len)
    
    # Ensure period is valid
    if period >= seq_len:
        period = max(3, (seq_len // 2) - 1)
        if period % 2 == 0:
            period -= 1
    
    try:
        # Apply STL decomposition
        stl = STL(ts_np, period=period, robust=robust)
        result = stl.fit()
        
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        
    except Exception as e:
        # Fallback: if STL fails, use simple moving average for trend
        print(f"STL decomposition failed: {e}. Using fallback method.")
        
        # Simple moving average for trend
        window = min(period, seq_len // 3)
        if window % 2 == 0:
            window += 1
        
        trend = np.convolve(ts_np, np.ones(window)/window, mode='same')
        # Handle edges
        half_window = window // 2
        for i in range(half_window):
            trend[i] = np.mean(ts_np[:i+half_window+1])
            trend[-(i+1)] = np.mean(ts_np[-(i+half_window+1):])
        
        # No seasonality in fallback
        seasonal = np.zeros_like(ts_np)
        
        # Residual is what's left
        residual = ts_np - trend - seasonal
    
    # Convert back to torch if needed
    if is_torch:
        trend = torch.from_numpy(trend).to(device).float()
        seasonal = torch.from_numpy(seasonal).to(device).float()
        residual = torch.from_numpy(residual).to(device).float()
    
    return trend, seasonal, residual


def stl_decompose_batch(x_ts, period=None, robust=True):
    """
    Apply STL decomposition to a batch of multivariate time series.
    
    Args:
        x_ts: Time series tensor of shape [batch_size, seq_len, n_channels]
        period: Period for STL decomposition. If None, auto-detected.
        robust: Whether to use robust STL
        
    Returns:
        trends: Trend components [batch_size, seq_len, n_channels]
        seasonals: Seasonal components [batch_size, seq_len, n_channels]
        residuals: Residual components [batch_size, seq_len, n_channels]
    """
    is_torch = isinstance(x_ts, torch.Tensor)
    if is_torch:
        device = x_ts.device
        batch_size, seq_len, n_channels = x_ts.shape
    else:
        batch_size, seq_len, n_channels = x_ts.shape
        device = None
    
    # Auto-detect period if not provided
    if period is None:
        period = auto_detect_period(seq_len)
    
    # Initialize output arrays
    if is_torch:
        trends = torch.zeros_like(x_ts)
        seasonals = torch.zeros_like(x_ts)
        residuals = torch.zeros_like(x_ts)
    else:
        trends = np.zeros_like(x_ts)
        seasonals = np.zeros_like(x_ts)
        residuals = np.zeros_like(x_ts)
    
    # Apply STL to each sample and channel
    for b in range(batch_size):
        for c in range(n_channels):
            if is_torch:
                ts = x_ts[b, :, c]
            else:
                ts = x_ts[b, :, c]
            
            trend, seasonal, residual = stl_decompose_single(ts, period, robust)
            
            if is_torch:
                trends[b, :, c] = trend
                seasonals[b, :, c] = seasonal
                residuals[b, :, c] = residual
            else:
                trends[b, :, c] = trend
                seasonals[b, :, c] = seasonal
                residuals[b, :, c] = residual
    
    return trends, seasonals, residuals


def reconstruct_from_components(trends, seasonals, residuals):
    """
    Reconstruct time series from STL components.
    
    Args:
        trends: Trend components
        seasonals: Seasonal components
        residuals: Residual components
        
    Returns:
        reconstructed: Reconstructed time series (trend + seasonal + residual)
    """
    return trends + seasonals + residuals


class STLComponentStorage:
    """
    Storage for STL components from the training set.
    Allows efficient random sampling during generation.
    """
    def __init__(self, device='cpu'):
        self.trends = []
        self.seasonals = []
        self.device = device
        self.is_finalized = False
        
    def add_batch(self, trends, seasonals):
        """
        Add a batch of trend and seasonal components.
        
        Args:
            trends: [batch_size, seq_len, n_channels]
            seasonals: [batch_size, seq_len, n_channels]
        """
        if self.is_finalized:
            raise RuntimeError("Cannot add to finalized storage")
        
        # Store as CPU tensors to save GPU memory
        if isinstance(trends, torch.Tensor):
            self.trends.append(trends.detach().cpu())
            self.seasonals.append(seasonals.detach().cpu())
        else:
            self.trends.append(torch.from_numpy(trends))
            self.seasonals.append(torch.from_numpy(seasonals))
    
    def finalize(self):
        """
        Finalize storage by concatenating all batches into single tensors.
        Call this after training loop is complete.
        """
        if len(self.trends) > 0:
            self.trends = torch.cat(self.trends, dim=0)
            self.seasonals = torch.cat(self.seasonals, dim=0)
            self.is_finalized = True
        else:
            raise RuntimeError("No components stored")
    
    def sample_random(self, batch_size):
        """
        Sample random trend and seasonal components.
        
        Args:
            batch_size: Number of samples to draw
            
        Returns:
            trends: [batch_size, seq_len, n_channels]
            seasonals: [batch_size, seq_len, n_channels]
        """
        if not self.is_finalized:
            raise RuntimeError("Storage must be finalized before sampling")
        
        n_stored = len(self.trends)
        indices = torch.randint(0, n_stored, (batch_size,))
        
        sampled_trends = self.trends[indices].to(self.device)
        sampled_seasonals = self.seasonals[indices].to(self.device)
        
        return sampled_trends, sampled_seasonals
    
    def __len__(self):
        """Return number of stored components"""
        if self.is_finalized:
            return len(self.trends)
        else:
            return sum(len(t) for t in self.trends)
    
    def clear(self):
        """Clear all stored components"""
        self.trends = []
        self.seasonals = []
        self.is_finalized = False


