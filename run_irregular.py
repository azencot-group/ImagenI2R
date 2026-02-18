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
from utils.utils_args import parse_args_irregular
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

        tst_config = {
            'feat_dim': args.input_size,
            'max_len': args.seq_len,
            'd_model': args.hidden_dim,
            'n_heads': args.n_heads,  # Number of attention heads
            'num_layers': args.num_layers,  # Number of transformer layers
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'pos_encoding': args.pos_encoding,  # or 'learnable'
            'activation': args.activation,
            'norm': args.norm,
            'freeze': args.freeze
        }
        # Initialize the TST model
        embedder = TSTransformerEncoder(
            feat_dim=tst_config['feat_dim'],
            max_len=tst_config['max_len'],
            d_model=tst_config['d_model'],
            n_heads=tst_config['n_heads'],
            num_layers=tst_config['num_layers'],
            dim_feedforward=tst_config['dim_feedforward'],
            dropout=tst_config['dropout'],
            pos_encoding=tst_config['pos_encoding'],
            activation=tst_config['activation'],
            norm=tst_config['norm'],
            freeze=tst_config['freeze']
        ).to(args.device)

        decoder = TST_Decoder(
            inp_dim=args.hidden_dim,
            hidden_dim=int(args.hidden_dim + (args.input_size - args.hidden_dim) / 2),
            layers=3,
            args=args
        ).to(args.device)
        optimizer_er = optim.Adam(chain(embedder.parameters(), decoder.parameters()))
        embedder.train()
        decoder.train()

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        best_disc_score = float('inf')

        print('logging_iter', args.logging_iter)
        for step in range(1, args.first_epoch + 1):
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                x = x[:, :, :-1]
                x = propagate_values(x)
                padding_masks = ~torch.isnan(x).any(dim=-1)
                h = embedder(x, padding_masks)

                # Decoder forward pass with time information
                x_tilde = decoder(h)

                x_no_nan = x[~torch.isnan(x)]
                x_tilde_no_nan = x_tilde[~torch.isnan(x)]
                loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)
                loss_e_0 = _loss_e_0(loss_e_t0)
                optimizer_er.zero_grad()
                loss_e_0.backward()
                optimizer_er.step()
                torch.cuda.empty_cache()

            print(
                "step: "
                + str(step)
                + "/"
                + str(args.first_epoch)
                + ", loss_e: "
                + str(np.round(np.sqrt(loss_e_t0.item()), 4))
            )


        for epoch in range(init_epoch, args.epochs):
            print("Starting epoch %d." % (epoch,))

            model.train()
            model.epoch = epoch

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                x = data[0].to(args.device)
                x_ts = x[:, :, :-1]

                x_ts = propagate_values(x_ts)
                padding_masks = ~torch.isnan(x_ts).any(dim=-1)
                x_img = model.ts_to_img(x_ts)
                mask = torch.isnan(x_img).float() * -1 + 1
                h = embedder(x_ts, padding_masks)
                x_recon = decoder(h)
                x_tilde_img = model.ts_to_img(x_recon)
                loss = model.loss_fn_irregular(x_tilde_img, mask)
                optimizer.zero_grad()
                if len(loss) == 2:
                    loss, to_log = loss
                    for key, value in to_log.items():
                        logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

                # #############Recovery######################
                h = embedder(x_ts, padding_masks)
                x_tilde = decoder(h)
                x_no_nan = x_ts[~torch.isnan(x_ts)]
                x_tilde_no_nan = x_tilde[~torch.isnan(x_ts)]
                loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)

                loss_e_0 = _loss_e_0(loss_e_t0)
                loss_e = loss_e_0
                optimizer_er.zero_grad()
                loss_e.backward()
                optimizer_er.step()
                torch.cuda.empty_cache()

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        for data in tqdm(test_loader):
                            # sample from the model
                            x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                            # --- convert to time series --
                            x_ts = model.img_to_ts(x_img_sampled)

                            gen_sig.append(x_ts.detach().cpu().numpy())
                            real_sig.append(data[0].detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)

                scores = evaluate_model_irregular(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

                # --- save checkpoint ---
                curr_disc_score = scores['disc_mean']

                # Du tu time consumption, we calculate predictive, fid and correlation only if we have an improvement in the disc score
                if curr_disc_score < best_disc_score:
                    new_scores = evaluate_model_irregular(real_sig, gen_sig, args, calc_other_metrics=True)

                    pred_score = new_scores['pred_score_mean']
                    fid_score = new_scores['fid_score_mean']
                    correlation_score = new_scores['correlation_score_mean']

                    best_disc_score = curr_disc_score
                    ema_model = model.model_ema if args.ema else None
                    save_checkpoint(args=args, our_model=model, our_optimizer=optimizer, ema_model=ema_model, encoder=embedder, decoder=decoder, tst_optimizer=optimizer_er, disc_score=best_disc_score, pred_score=pred_score, fid_score=fid_score, correlation_score=correlation_score)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_irregular()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
