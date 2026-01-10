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
from utils.utils_stl import stl_decompose_batch
from models.stl_synthesis import STLSynthesisModel
from models.sampler import DiffusionProcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        # Initialize Synthesis Model
        model = STLSynthesisModel(args=args, device=args.device).to(args.device)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        state = dict(model=model, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)

        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))

            model.train()
            model.epoch = epoch

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                x_ts = x[:, :, :-1]  # (B, T, C)

                # === STL DECOMPOSITION ===
                # Extract ground truth components for multi-head supervision
                trends_gt, seasonals_gt, residuals_gt = stl_decompose_batch(x_ts, period=None, robust=True)
                
                # === NOISING ===
                # Noise the FULL signal to train synthesis from noisy observations
                rnd_normal = torch.randn([x_ts.shape[0], 1, 1], device=x_ts.device)
                sigma = (rnd_normal * model.P_std + model.P_mean).exp()
                noise = torch.randn_like(x_ts) * sigma
                x_ts_noisy = x_ts + noise
                
                # === SYNTHESIS HEADS ===
                # Synthesize trend and seasonality from noisy input
                trend_pred, season_pred = model.synthesize_components(x_ts_noisy)
                
                # === RESIDUAL HEAD (IMAGE-BASED) ===
                # Predict residual image using the same U-Net logic as run_regular_stl.py
                x_img_noisy = model.ts_to_img(x_ts_noisy)
                res_img_pred = model.net(x_img_noisy, sigma.view(-1, 1, 1, 1))
                
                # Target residual image from ground truth STL residual
                res_img_gt = model.ts_to_img(residuals_gt)
                
                # === LOSS CALCULATION ===
                # 1. Karras loss on residuals (matching run_regular_stl.py)
                weight = (sigma ** 2 + model.sigma_data ** 2) / (sigma * model.sigma_data) ** 2
                loss_res = (weight.view(-1, 1, 1, 1) * (res_img_pred - res_img_gt).square()).mean()
                
                # 2. Synthesis loss (matching Diffusion-TS inspiration)
                loss_trend = F.mse_loss(trend_pred, trends_gt)
                loss_season = F.mse_loss(season_pred, seasonals_gt)
                
                total_loss = loss_res + 0.5 * loss_trend + 0.5 * loss_season
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()
                
                if i % 10 == 0:
                    logger.log('train/total_loss', total_loss.item(), epoch)
                    logger.log('train/res_loss', loss_res.item(), epoch)
                    logger.log('train/trend_loss', loss_trend.item(), epoch)
                    logger.log('train/season_loss', loss_season.item(), epoch)

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        # Configure DiffusionProcess to work directly with time series shape (B, T, C)
                        # The model's forward() handles reconstruction internally
                        process = DiffusionProcess(
                            args, model, 
                            img_to_ts=lambda x: x, 
                            ts_to_img=lambda x: x,
                            shape=(args.seq_len, args.input_channels)
                        )
                        
                        for data in tqdm(test_loader):
                            batch_size = data[0].shape[0]
                            # sampling() returns reconstructed x0 estimates
                            x_sampled = process.sampling(sampling_number=batch_size)
                            
                            gen_sig.append(x_sampled.detach().cpu().numpy())
                            real_sig.append(data[0].detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)

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


