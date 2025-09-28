from metrics.discriminative_torch import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.context_fid import calculate_fid
from metrics.correlation_score import calculate_pearson_correlation
import numpy as np
import torch

def evaluate_model_irregular(real_sig, gen_sig, args, calc_other_metrics=False):
    """
    Args:
        real_sig: real signal
        gen_sig: generated signal
        args: args
        calc_other_metrics: in case we want to calculate predictive, fid, and correlation
    """

    # proceed with short term evaluation
    metric_iteration = 1

    ## for deterministic results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    disc_res = []

    if calc_other_metrics:
        predictive_score = list()
        for _ in range(metric_iteration):
            temp_pred = predictive_score_metrics(real_sig, gen_sig)
            predictive_score.append(temp_pred)
        pred_mean, pred_std = np.round(np.mean(predictive_score), 4), np.round(np.std(predictive_score), 4)
        print('predictive_score_mean: {}, predictive_score_std: {}'.format(pred_mean, pred_std))

        fid_mean, fid_std, fid_conf_interval = calculate_fid(real_sig, gen_sig)
        print('fid_score_mean: {}, fid_score_std: {}, fid_score_conf_interval: {}'.format(fid_mean, fid_std, fid_conf_interval))

        correlation_score_mean, correlation_score_std, correlation_score_conf_interval = calculate_pearson_correlation(
            real_sig, gen_sig)
        print('correlation_score_mean: {}, correlation_score_std: {}, correlation_score_conf_interval: {}'.format(
                correlation_score_mean, correlation_score_std, correlation_score_conf_interval))

        return {'pred_score_mean': pred_mean, 'pred_score_std': pred_std, 'fid_score_mean': fid_mean, 'fid_score_std': fid_std,
                'fid_score_conf_interval': fid_conf_interval, 'correlation_score_mean': correlation_score_mean,
                'correlation_score_std': correlation_score_std,
                'correlation_score_conf_interval': correlation_score_conf_interval}

    else:
        for _ in range(metric_iteration):
            dsc = discriminative_score_metrics(real_sig, gen_sig, args)
            disc_res.append(dsc)
        disc_mean, disc_std = np.round(np.mean(disc_res), 4), np.round(np.std(disc_res), 4)
        print('disc_score mean: {}, std: {}'.format(disc_mean, disc_std))
        return {'disc_mean': disc_mean, 'disc_std': disc_std}
