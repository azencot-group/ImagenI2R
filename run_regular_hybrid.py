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
from models.hybrid import HybridDiffusionModel
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

        # Initialize Hybrid Model
        model = HybridDiffusionModel(args=args, device=args.device).to(args.device)

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
                # Extract ground truth components for training matching
                trends_gt, seasonals_gt, residuals_gt = stl_decompose_batch(x_ts, period=None, robust=True)
                
                # === HYBRID TRAINING ===
                # Train the residual part like run_regular_stl.py
                # Train the trend/seasonality part via synthesis matching
                loss, to_log = model.loss_fn_hybrid(x_ts, trends_gt, seasonals_gt, residuals_gt)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()
                
                if i % 10 == 0:
                    for key, value in to_log.items():
                        logger.log(f'train/{key}', value, epoch)

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        # Configure DiffusionProcess to work directly with time series shape
                        # The Hybrid model handles image/TS conversions internally
                        process = DiffusionProcess(
                            args, model, 
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


