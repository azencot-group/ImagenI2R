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
from utils.utils_args import parse_args_irregular_transformer
from models.transformer_interpretable_edm_irregular import TSTransformerInterpretableEDMIrregular
from models.sampler import DiffusionProcess

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

        # Initialize the Interpretable Transformer EDM Irregular model
        model = TSTransformerInterpretableEDMIrregular(args=args, device=args.device).to(args.device)

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
                # Remove index column from irregular data (the last column)
                if x.shape[-1] > args.input_channels:
                    x_ts = x[:, :, :-1]
                else:
                    x_ts = x

                loss, to_log = model.loss_fn_irregular(x_ts)
                
                optimizer.zero_grad()
                for key, value in to_log.items():
                    logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        # DiffusionProcess for TS domain sampling
                        process = DiffusionProcess(args, model.net, model.img_to_ts, model.ts_to_img,
                                                   (args.seq_len, args.input_channels))
                        for data in tqdm(test_loader, desc='Generating samples'):
                            # data[0] is clean TS: (B, T, C)
                            clean_ts = data[0].to(args.device)
                            
                            # sample from the model directly in TS domain
                            # For irregular, we are generating full time series from noise
                            x_ts_sampled = process.sampling(sampling_number=clean_ts.shape[0])
                            
                            gen_sig.append(x_ts_sampled.detach().cpu().numpy())
                            real_sig.append(clean_ts.detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)

                # evaluate_model_irregular takes (real, gen, args)
                scores = evaluate_model_irregular(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_irregular_transformer()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)

