import torch
import torch.multiprocessing
from torch import optim
import torch.nn.functional as F
import os, sys
import glob
import numpy as np
import logging
from tqdm import tqdm
from itertools import chain

from metrics import evaluate_model_irregular
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from utils.utils import restore_state, create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_irregular, parse_args_regular
from utils.utils_stl import stl_decompose_batch, reconstruct_from_components, STLComponentStorage
from models.our import TS2img_Karras
from models.sampler import DiffusionProcess
from models.decoder import TST_Decoder
from models.TST import TSTransformerEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')

def propagate_values_forward(tensor):
    # Iterate over the batch and channels
    for b in range(tensor.size(0)):
            # Extract the sequence for the current batch and channel
            sequence = tensor[b]
            if torch.isnan(sequence).all():
                if b + 1 < tensor.size(0):
                    tensor[b] = tensor[b + 1]
                else:
                    tensor[b] = tensor[b - 1]
    return tensor

def propagate_values(tensor):
    tensor = propagate_values_forward(tensor)
    return tensor

def save_checkpoint(args, our_model, our_optimizer, ema_model, encoder, decoder, tst_optimizer, disc_score, pred_score=None, fid_score=None, correlation_score=None):
    """
    Saves the model checkpoint to the specified directory based on args and disc_score.
    """
    try:
        main_path = args.model_save_path
        seq_len = args.seq_len
        data_set_name = args.dataset
        missing_rate = int(args.missing_rate * 100)

        # Build the directory structure
        full_path = os.path.join(main_path, f'seq_len_{seq_len}', data_set_name, f'missing_rate_{missing_rate}')
        os.makedirs(full_path, exist_ok=True)

        # ---- Remove old files ----
        for f in glob.glob(os.path.join(full_path, "*")):
            try:
                os.remove(f)
            except IsADirectoryError:
                # if subdirectories might exist, handle recursively
                import shutil
                shutil.rmtree(f)

        # Generate the file name
        filename = f"disc_score_{disc_score}_pred_score_{pred_score}_fid_score_{fid_score}_correlation_score_{correlation_score}.pth"

        filepath = os.path.join(full_path, filename)

        # Save the checkpoint
        torch.save({
            'our_model_state_dict': our_model.state_dict(),
            'our_optimizer_state_dict': our_optimizer.state_dict(),
            'ema_model': ema_model.state_dict(),
            'tst_encoder': encoder.state_dict(),
            'tst_decoder': decoder.state_dict(),
            'tst_optimizer': tst_optimizer.state_dict(),
            'disc_score': disc_score,
            'pred_score': pred_score,
            'fid_score': fid_score,
            'correlation_score': correlation_score,
            'args': vars(args)
        }, filepath)

        print(f"Checkpoint saved at: {filepath}")

    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

def _loss_e_t0(x_tilde, x):
    return F.mse_loss(x_tilde, x)

def _loss_e_0(loss_e_t0):
    return torch.sqrt(loss_e_t0) * 10


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

        model = TS2img_Karras(args=args, device=args.device).to(args.device)

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

        # Initialize STL component storage
        stl_storage = STLComponentStorage(device=args.device)
        logging.info("Initialized STL component storage for trend and seasonality.")

        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))

            model.train()
            model.epoch = epoch

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                x_ts = x[:, :, :-1]  # Remove time index column

                # === STL DECOMPOSITION ===
                # Decompose time series into trend, seasonal, and residual components
                trends, seasonals, residuals = stl_decompose_batch(x_ts, period=None, robust=True)
                
                # Store trend and seasonal components for later sampling
                # Only store during first epoch to avoid duplicates
                if epoch == init_epoch:
                    stl_storage.add_batch(trends, seasonals)
                
                # === TRAIN ON RESIDUALS ONLY ===
                # Convert residual component to image representation
                x_tilde_img = model.ts_to_img(residuals)
                
                # Compute loss on residual images
                loss = model.loss_fn_regular(x_tilde_img)
                optimizer.zero_grad()
                if len(loss) == 2:
                    loss, to_log = loss
                    for key, value in to_log.items():
                        logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

            # Finalize storage after first epoch
            if epoch == init_epoch and not stl_storage.is_finalized:
                stl_storage.finalize()
                logging.info(f"STL storage finalized. Stored {len(stl_storage)} trend/seasonal component pairs.")

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net, model.img_to_ts, model.ts_to_img,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        for data in tqdm(test_loader):
                            batch_size = data[0].shape[0]
                            
                            # === SAMPLE RESIDUALS FROM DIFFUSION MODEL ===
                            x_img_sampled = process.sampling(sampling_number=batch_size)
                            
                            # Convert sampled residual images back to time series
                            residuals_sampled = model.img_to_ts(x_img_sampled)
                            
                            # === RANDOMLY SELECT TREND AND SEASONALITY ===
                            # Sample random trend and seasonal components from training set
                            sampled_trends, sampled_seasonals = stl_storage.sample_random(batch_size)
                            
                            # === RECONSTRUCT TIME SERIES ===
                            # Combine: x = trend + seasonal + residual
                            x_ts_reconstructed = reconstruct_from_components(
                                sampled_trends, sampled_seasonals, residuals_sampled
                            )

                            gen_sig.append(x_ts_reconstructed.detach().cpu().numpy())
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


