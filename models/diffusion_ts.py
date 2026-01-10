import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, reduce, repeat


# --- Utilities ---

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.transpose(*self.shape)

class GELU2(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)
    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )
    def forward(self, x):
        return self.sequential(x).transpose(1, 2)

# --- Decomposition Blocks ---

class TrendBlock(nn.Module):
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )
        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals

class FourierLayer(nn.Module):
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t, device=x.device)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t, device=x.device)[self.low_freq:]
        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        time_steps = rearrange(torch.arange(t, dtype=torch.float, device=x_freq.device), 't -> () () t ()')
        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * time_steps + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = max(1, int(self.factor * math.log(length)))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0), device=x_freq.device), 
                                        torch.arange(x_freq.size(2), device=x_freq.device), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple

# --- Transformer Architecture ---

class FullAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, att.mean(dim=1)

class CrossAttention(nn.Module):
    def __init__(self, n_embd, condition_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, att.mean(dim=1)

class EncoderBlock(nn.Module):
    def __init__(self, n_embd=1024, n_head=16, attn_pdrop=0.1, resid_pdrop=0.1, mlp_hidden_times=4, activate='GELU'):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        act = nn.GELU() if activate == 'GELU' else GELU2()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att

class Encoder(nn.Module):
    def __init__(self, n_layer=14, n_embd=1024, n_head=16, attn_pdrop=0., resid_pdrop=0., mlp_hidden_times=4, block_activate='GELU'):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(
            n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times, activate=block_activate
        ) for _ in range(n_layer)])
    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block in self.blocks:
            x, _ = block(x, t, mask=padding_masks, label_emb=label_emb)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, n_channel, n_feat, n_embd=1024, n_head=16, attn_pdrop=0.1, resid_pdrop=0.1, mlp_hidden_times=4, activate='GELU', condition_dim=1024):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn1 = FullAttention(n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.attn2 = CrossAttention(n_embd=n_embd, condition_embd=condition_dim, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.ln1_1 = AdaLayerNorm(n_embd)
        act = nn.GELU() if activate == 'GELU' else GELU2()
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        self.seasonal = FourierLayer(d_model=n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )
        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x.transpose(1, 2)).chunk(2, dim=1) # (B, C*2, T) -> (B, C, T), (B, C, T)
        trend, season = self.trend(x1), self.seasonal(x2.transpose(1, 2))
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season

class Decoder(nn.Module):
    def __init__(self, n_channel, n_feat, n_embd=1024, n_head=16, n_layer=10, attn_pdrop=0.1, resid_pdrop=0.1, mlp_hidden_times=4, block_activate='GELU', condition_dim=512):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.ModuleList([DecoderBlock(
            n_feat=n_feat, n_channel=n_channel, n_embd=n_embd, n_head=n_head,
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, mlp_hidden_times=mlp_hidden_times,
            activate=block_activate, condition_dim=condition_dim
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, t_len, _ = x.shape
        season = torch.zeros((b, t_len, self.n_feat), device=x.device)
        trend = torch.zeros((b, t_len, self.n_feat), device=x.device)
        mean = []
        for block in self.blocks:
            x, residual_mean, residual_trend, residual_season = block(x, enc, t, mask=padding_masks, label_emb=label_emb)
            season = season + residual_season
            trend = trend + residual_trend
            mean.append(residual_mean)
        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season

class DiffusionTS_Transformer(nn.Module):
    def __init__(self, n_feat, n_channel, n_layer_enc=5, n_layer_dec=10, n_embd=256, n_heads=8,
                 attn_pdrop=0.1, resid_pdrop=0.1, mlp_hidden_times=4, block_activate='GELU', max_len=1024, **kwargs):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)
        self.combine_s = nn.Conv1d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)
        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

    def forward(self, input, t, padding_masks=None, return_res=False):
        # input: (B, T, C)
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)
        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        # Reconstruct components
        s_comp = self.combine_s(season.transpose(1, 2)).transpose(1, 2)
        t_comp = self.combine_m(mean).transpose(1, 2) + res_m + trend
        r_comp = res - res_m
        
        if return_res:
            return t_comp, s_comp, r_comp
        return t_comp, s_comp + r_comp

    # Compatibility methods for the pipeline
    def round_sigma(self, sigma):
        return sigma

    @property
    def sigma_min(self):
        return 0.002
    
    @property
    def sigma_max(self):
        return 80.0


