import torch
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
import os, sys
import glob
import numpy as np
import logging
from tqdm import tqdm

from metrics import evaluate_model_irregular
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_regular
from models.diffusion_ts import DiffusionTS_Transformer
from models.sampler import DiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

# --- Model Wrapper for DiffusionProcess Compatibility ---

class DiffusionTS_Wrapper(torch.nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.transformer = transformer_model
        # DiffusionProcess expects these properties
        self.sigma_min = transformer_model.sigma_min
        self.sigma_max = transformer_model.sigma_max

    def forward(self, x, sigma, labels=None, augment_labels=None):
        # x is (B, C, H, W) if coming from DiffusionProcess normally, 
        # but we configure DiffusionProcess to use (B, T, C)
        # transformer expects (B, T, C)
        # sigma is (B, 1, 1, 1) or similar
        
        # Diffusion-TS architecture predicts x0 directly
        t_comp, residual_comp = self.transformer(x, sigma.view(-1))
        x0_hat = t_comp + residual_comp
        return x0_hat

    def round_sigma(self, sigma):
        return self.transformer.round_sigma(sigma)

# --- Fourier Loss Function ---

def fourier_loss(x, x_hat):
    # x, x_hat: (B, T, C)
    x_fft = torch.fft.rfft(x, dim=1)
    x_hat_fft = torch.fft.rfft(x_hat, dim=1)
    
    loss_mse = F.mse_loss(x, x_hat)
    loss_fft = F.mse_loss(torch.abs(x_fft), torch.abs(x_hat_fft)) + \
               F.mse_loss(torch.angle(x_fft), torch.angle(x_hat_fft))
    
    return loss_mse + 0.1 * loss_fft

def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up neptune logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        # Initialize Diffusion-TS Transformer
        # n_feat = input_channels, n_channel = seq_len
        transformer = DiffusionTS_Transformer(
            n_feat=args.input_channels, 
            n_channel=args.seq_len,
            max_len=args.seq_len
        ).to(args.device)
        
        model_wrapper = DiffusionTS_Wrapper(transformer)

        # optimizer
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        state = dict(model=transformer, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            init_epoch = restore_state(args, state)

        # print model parameters
        print_model_params(logger, transformer)

        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))

            transformer.train()
            transformer.epoch = epoch

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                x_ts = x[:, :, :-1] # (B, T, C)

                # Diffusion step sampling
                # EDM / Karras style sigma sampling
                P_mean = -1.2
                P_std = 1.2
                rnd_normal = torch.randn([x_ts.shape[0], 1, 1], device=x_ts.device)
                sigma = (rnd_normal * P_std + P_mean).exp()
                
                # Add noise
                noise = torch.randn_like(x_ts) * sigma
                noised_x = x_ts + noise
                
                # Predict x0 through decomposition
                # Forward pass returns (trend, season_error)
                t_comp, residual_comp = transformer(noised_x, sigma.view(-1))
                x0_hat = t_comp + residual_comp
                
                # Loss computation: Component-aware + Fourier
                loss = fourier_loss(x_ts, x0_hat)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.)
                optimizer.step()
                
                if i % 10 == 0:
                    logger.log('train/loss', loss.item(), epoch)

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                transformer.eval()
                with torch.no_grad():
                    # We configure DiffusionProcess to work directly with time series shape
                    # No image transformations needed here as the model is a Transformer
                    process = DiffusionProcess(
                        args, model_wrapper, 
                        img_to_ts=lambda x: x, 
                        ts_to_img=lambda x: x,
                        shape=(args.seq_len, args.input_channels)
                    )
                    
                    for data in tqdm(test_loader):
                        batch_size = data[0].shape[0]
                        # sample from the model - this synthesizes T, S, and R internally
                        x_sampled = process.sampling(sampling_number=batch_size)
                        
                        gen_sig.append(x_sampled.detach().cpu().numpy())
                        real_sig.append(data[0].detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)

                # evaluate
                scores = evaluate_model_irregular(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_regular()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)


