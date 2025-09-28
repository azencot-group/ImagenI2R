import torch
import torch.nn as nn
from contextlib import contextmanager
from models.networks import EDMPrecond
from models.ema import LitEma
from models.img_transformations import DelayEmbedder

class TS2img_Karras(nn.Module):
    def __init__(self, args, device):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.net = EDMPrecond(args.img_resolution, args.input_channels, channel_mult=args.ch_mult,
                              model_channels=args.unet_channels, attn_resolutions=args.attn_resolution)

        self.delay = args.delay
        self.embedding = args.embedding
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.num_features = args.input_channels

        self.ts_img = DelayEmbedder(self.device, args.seq_len, args.delay, args.embedding, self.batch_size, self.num_features)

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def ts_to_img(self, signal):
        """
        Args:
            signal: signal to convert to image
        """
        return self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    def loss_fn_irregular(self, x, mask=None):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        if mask is None:
            mask = torch.isnan(x).float() * -1 + 1
            x = torch.nan_to_num(x, nan=0.0)
        output, weight = self.forward_irregular(x, mask)
        x = self.unpad(x * mask, x.shape)
        output = self.unpad(output * mask, x.shape)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()
        return loss, to_log

    def forward_irregular(self, x, mask, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma
        masked_noise = n * (mask)
        D_yn = self.net(y + masked_noise, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to temporarily switch to EMA weights during inference.
        Args:
            context: some string to print when switching to EMA weights

        Returns:

        """
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        """
        this function updates the EMA model, if it is used
        Args:
            *args:

        Returns:

        """
        if self.use_ema:
            self.model_ema(self.net)
