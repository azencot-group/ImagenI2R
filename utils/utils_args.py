from omegaconf import OmegaConf
import argparse

def parse_args_irregular():
    """
    Parse arguments for unconditional models
    Returns: unconditioanl generation args namespace

    """
    parser = argparse.ArgumentParser()
    # --- general ---
    # NOTE: the following arguments are general, they are not present in the config file:
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use for dataloader')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--neptune', type=bool, default=False, help='use neptune logger')
    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument('--tags', type=str, default=['30 missing rate'], help='tags for neptune logger', nargs='+')

    # --- diffusion process --- #
    parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic sampling')

    # ## --- config file --- # ##
    # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
    parser.add_argument('--config', type=str, default='./configs/seq_len_24/stock.yaml', help='config file')
    parser.add_argument('--model_save_path', type=str, default='./saved_models', help='path to save the model')


    # --- training ---
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')

    # --- data ---:
    parser.add_argument('--dataset',
                        choices=['sine', 'energy', 'mujoco', 'stock', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco)')

    # --- image transformations ---:
    parser.add_argument('--delay', type=int,
                        help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int,
                        help='embedding for the delay embedding transformation, only needed if using delay embedding')

    # --- model--- :
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int,
                        help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')

    # --- TST ---
    parser.add_argument('--hidden_dim', type=int, default=40, help='dimension of the hidden layer')
    parser.add_argument('--r_layer', type=int, default=2, help='number of RNN layers')
    parser.add_argument('--last_activation_r', type=str, default='sigmoid', help='last activation function for RNN layers')
    parser.add_argument('--first_epoch', type=int, default=2, help='number of first epoch to start training')
    parser.add_argument('--x_hidden', type=int, default=48, help='dimension of x hidden layer')
    parser.add_argument('--input_size', type=int, default=1, help='input size of the model')

    # Adding new arguments for tst_config
    parser.add_argument('--n_heads', type=int, default=5, help='number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of feedforward layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--pos_encoding', type=str, choices=['fixed', 'learnable'], default='fixed',
                        help='positional encoding type')
    parser.add_argument('--activation', type=str, choices=['relu', 'gelu'], default='gelu', help='activation function')
    parser.add_argument('--norm', type=str, choices=['BatchNorm', 'LayerNorm'], default='BatchNorm',
                        help='normalization type')
    parser.add_argument('--freeze', type=bool, default=False, help='freeze transformer layers')

    parser.add_argument('--ts_rate', type=float, default=0, help='teacher forcing rate for tst')
    parser.add_argument('--save_model', type=bool, default=False, help='save model')
    parser.add_argument('--gaussian_noise_level', type=float, default=0.0, help='noise level injected to the original data')
    parser.add_argument('--new_metrics', type=int, default=1, help='save model')

    # --- logging ---s
    parser.add_argument('--logging_iter', type=int, default=10, help='number of iterations between logging')
    parser.add_argument('--percent', type=int, default=100)
    parsed_args = parser.parse_args()

    # load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    # override config file with command line args
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    # add to the parsed args, configs that are not in the parsed args but do in the config file
    # this is needed since multiple config files setups may be used
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)

    parsed_args.input_size = parsed_args.input_channels
    return parsed_args
