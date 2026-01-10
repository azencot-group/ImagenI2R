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
from models.transformer_interpretable import TSTransformerInterpretable
from models.ema import LitEma

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader, _ = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        # Initialize the Interpretable Transformer Diffusion model
        model = TSTransformerInterpretable(args=args, device=args.device)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # EMA initialization
        if args.ema:
            use_ema = True
            model_ema = LitEma(model, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            use_ema = False

        state = dict(model=model, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_param = model_ema if use_ema else None
            init_epoch = restore_state(args, state, ema_model=ema_param)

        # print model parameters
        print_model_params(logger, model)


        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))

            model.train()
            model.epoch = epoch

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                # Remove index column from irregular data if present
                if x.shape[-1] > args.input_channels:
                    x_ts = x[:, :, :-1]
                else:
                    x_ts = x

                # DDPM style training
                loss, to_log = model.loss_fn_regular(x_ts)
                
                optimizer.zero_grad()
                for key, value in to_log.items():
                    logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                
                if use_ema:
                    model_ema(model)

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                
                # Evaluation logic
                # We want to use EMA weights if available
                if use_ema:
                    model_ema.store(model.parameters())
                    model_ema.copy_to(model)
                
                with torch.no_grad():
                    for data in tqdm(test_loader, desc='Generating samples'):
                        # data[0] is clean TS: (B, T, C)
                        clean_ts = data[0].to(args.device)
                        
                        # sample from the model directly in TS domain
                        x_ts_sampled = model.sample_ts(batch_size=clean_ts.shape[0])
                        
                        gen_sig.append(x_ts_sampled.detach().cpu().numpy())
                        real_sig.append(clean_ts.detach().cpu().numpy())

                if use_ema:
                    model_ema.restore(model.parameters())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)

                # evaluate_model_irregular takes (real, gen, args)
                scores = evaluate_model_irregular(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_regular()
    # Override some args for this specific setup if they are None in config
    if args.diffusion_steps is None:
        args.diffusion_steps = 200 # Default for Diffusion-TS
    
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)

