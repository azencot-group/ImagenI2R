import os
import logging

def log_config_and_tags(args, logger, name):
    logger.log_name_params('config/hyperparameters', vars(args))
    logger.log_name_params('config/name', name)
    logger.add_tags(args.tags)
    logger.add_tags([args.dataset])


def create_model_name_and_dir(args):
    name = (f'conditional-'
            f'bs={args.batch_size}-'
            f'-lr={args.learning_rate:.4f}-'
            f'ch_mult={args.ch_mult}-'
            f'attn_res={args.attn_resolution}-'
            f'unet_ch={args.unet_channels}'
            )

    assert (args.delay is not None and args.embedding is not None)
    name += f'-delay={args.delay}-{args.embedding}'
    args.log_dir = '%s/%s/%s' % (args.log_dir, args.dataset, name)
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    return name

def restore_state(args, state,ema_model=None):
    logging.info("restoring checkpoint from: {}".format(args.log_dir))
    restore_checkpoint(args.log_dir, state, ema_model = ema_model)
    init_epoch = state['epoch']
    return init_epoch


def print_model_params(logger, model):
    params_num = sum(param.numel() for param in model.parameters())
    logging.info("number of model parameters: {}".format(params_num))
    logger.log_name_params('config/params_num', params_num)

