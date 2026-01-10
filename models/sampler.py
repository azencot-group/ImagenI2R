import numpy as np
import torch

class DiffusionProcess():
    def __init__(self, args, diffusion_fn, img_to_ts, ts_to_img, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''
        self.args = args
        self.device = args.device
        self.shape = shape
        self.img_to_ts = img_to_ts
        self.ts_to_img = ts_to_img
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps), dim=0).to(device=self.device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=self.device), self.alpha_bars[:-1]])
        self.deterministic = args.deterministic
        self.net = diffusion_fn.to(device=self.device)
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.S_churn = 0
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.num_steps = args.diffusion_steps

    def sample(self, latents, class_labels=None):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    @torch.no_grad()
    def sampling(self, sampling_number=16, impute=False, xT=None):
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        return self.sample(xT)

    def sample_ambient(self, latents, mask, class_labels=None):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0] * mask
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_to_impute = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)) * mask
            x_cur = x_cur * (1 - mask)
            x_hat = x_cur + x_to_impute

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            if i < self.num_steps - 1:
                x_next = x_hat + (t_next - t_hat) * d_cur * mask
            else:
                x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) * mask

        return x_next

    @torch.no_grad()
    def sampling_ambient(self, missing_rate=0.7, extra_corruption_rate=0.1, sampling_number=16, xT=None):
        '''
        Ambient diffusion sampling: generates samples using a fixed random mask.
        missing_rate: original missing rate (e.g., 0.7 = 70% missing)
        extra_corruption_rate: extra corruption rate applied to valid pixels (e.g., 0.1 = 10%)
        sampling_number: number of samples to generate
        xT: optional initial noise (if None, will generate random noise)
        '''
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        
        # Generate random mask: 0 for missing, 1 for valid
        original_mask = (torch.rand_like(xT) > missing_rate).float()
        transformation_mask = self.ts_to_img(self.img_to_ts(torch.ones([sampling_number, *self.shape]).to(device=self.device)))
        original_mask = original_mask * transformation_mask
        # Apply extra corruption: corrupt 10% of valid pixels
        extra_corruption_mask = (torch.rand_like(xT) < extra_corruption_rate) & (original_mask == 1)
        
        # Combined mask: 0 where missing OR extra corrupted
        combined_mask = original_mask * (~extra_corruption_mask).float()
        
        return self.sample_ambient(xT, combined_mask)

    @torch.no_grad()
    def sampling_ambient_2(self, missing_rate=0.7, sampling_number=16, xT=None):
        '''
        Ambient diffusion sampling: generates samples using a fixed random mask.
        missing_rate: original missing rate (e.g., 0.7 = 70% missing)
        extra_corruption_rate: extra corruption rate applied to valid pixels (e.g., 0.1 = 10%)
        sampling_number: number of samples to generate
        xT: optional initial noise (if None, will generate random noise)
        '''
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        mask = self.generate_irregular_mask((sampling_number, 24, 6), 0.4)
        mask_img = self.ts_to_img(mask)
        mask_img = torch.isnan(mask_img).float() * -1 + 1
        # x_img = torch.zeros(sampling_number, 6, self.args.img_resolution, self.args.img_resolution).to(self.args.device)
        return self.sample_ambient(xT, mask_img)

    def generate_irregular_mask(self, data_shape, missing_rate, seed=None):
        """
        Generate a random mask for irregular/missing data.
        """
        if seed is not None:
            torch.manual_seed(seed)
        batch_size, seq_len, num_features = data_shape
        mask = torch.ones((batch_size, seq_len, num_features), dtype=torch.float64)
        missing_per_seq = int(seq_len * missing_rate)

        for i in range(batch_size):
            missing_indices = torch.randperm(seq_len)[:missing_per_seq]
            mask[i, missing_indices, :] = float('nan')

        return mask

    def sample_irregular_ambient(self, latents, original_mask, class_labels=None):
        """
        Sample for irregular ambient diffusion training.
        This method generates complete regular time series from pure noise,
        aligned with loss_fn_irregular_ambient_2 training.
        
        Args:
            latents: initial noise tensor
            original_mask: mask indicating original valid pixels (1) vs original missing pixels (0)
            class_labels: optional class labels
        
        Returns:
            Denoised complete time series
        """
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop - denoise only observed pixels
        x_next = latents.to(torch.float64) * t_steps[0] * original_mask
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)

            # Add noise only to the valid pixels
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur) * original_mask

            # Euler step - network predicts denoised output for observed pixels
            # We mask input to match training distribution: noisy observed + zeroed missing
            net_input = x_hat * original_mask
            denoised = self.net(net_input, t_hat, class_labels).to(torch.float64)
            
            # Update only the observed pixels. 
            # Since net was only trained on observed pixels (loss_fn_irregular_ambient_2), 
            # its predictions on missing pixels are not reliable.
            d_cur = (x_hat - denoised) / t_hat
            x_next = (x_hat + (t_next - t_hat) * d_cur) * original_mask

            # Apply 2nd order correction, also masked.
            if i < self.num_steps - 1:
                net_input_next = x_next * original_mask
                denoised = self.net(net_input_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = (x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)) * original_mask

        return x_next

    @torch.no_grad()
    def sampling_irregular_ambient(self, missing_rate=0.7, sampling_number=16, xT=None):
        """
        Wrapper for irregular ambient sampling aligned with loss_fn_irregular_ambient_2.
        Generates complete regular time series from irregular observations.
        
        Args:
            missing_rate: rate of missing data in the original irregular time series
            sampling_number: number of samples to generate
            xT: optional initial noise
        
        Returns:
            Complete regular time series samples
        """
        if xT is None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        
        # Generate original mask (simulating the irregular pattern the model learned from)
        mask_ts = self.generate_irregular_mask((sampling_number, 24, 6), missing_rate)
        original_mask = self.ts_to_img(mask_ts)
        original_mask = torch.isnan(original_mask).float() * -1 + 1  # 1 for valid, 0 for missing

        mask_ts_extra = self.generate_irregular_mask((sampling_number, 24, 6), 0.2)
        extra_mask = self.ts_to_img(mask_ts_extra)
        extra_mask = torch.isnan(extra_mask).float() * -1 + 1  # 1 for valid, 0 for missing

        combined_mask = extra_mask * original_mask
        
        return self.sample_irregular_ambient(xT, combined_mask)