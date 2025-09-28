"""
Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py (PyTorch conversion)

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error

def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


class GRUPredictorIrregular(nn.Module):
    """
    This class replicates the same 'predictor' logic used by TensorFlow's
    dynamic_rnn(GRUCell) + dense layer + sigmoid for the irregular case
    (input_size = dim, output_size = dim).
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Single-layer GRU, analogous to a TF GRUCell inside dynamic_rnn
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          batch_first=True)
        # Equivalent to tf.layers.dense(..., units=dim)
        self.dense = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, T):
        """
        Emulates tf.nn.dynamic_rnn(..., sequence_length=T):
          - We pack the input according to T
          - Run the GRU
          - Unpack (pad_packed_sequence) so that final outputs
            align with the original shape [batch_size, max_seq_len-1, dim]
          - Then pass through a dense layer and sigmoid
        """
        # Convert lists of lengths T to a tensor
        lengths_tensor = torch.tensor(T, dtype=torch.long)
        # Pack the sequence
        packed_X = nn.utils.rnn.pack_padded_sequence(X,
                                                     lengths=lengths_tensor,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        # GRU forward
        p_outputs, _ = self.gru(packed_X)
        # Unpack
        p_outputs, _ = nn.utils.rnn.pad_packed_sequence(p_outputs,
                                                        batch_first=True)
        # Dense + Sigmoid
        y_hat_logit = self.dense(p_outputs)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat


class GRUPredictorShortTerm(nn.Module):
    """
    This class replicates the same 'predictor' logic for the short-term case
    (input_size = dim-1, output_size = 1).
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          batch_first=True)
        self.dense = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, T):
        lengths_tensor = torch.tensor(T, dtype=torch.long)
        packed_X = nn.utils.rnn.pack_padded_sequence(X,
                                                     lengths=lengths_tensor,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        p_outputs, _ = self.gru(packed_X)
        p_outputs, _ = nn.utils.rnn.pad_packed_sequence(p_outputs,
                                                        batch_first=True)
        y_hat_logit = self.dense(p_outputs)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat


def predictive_score_metrics(ori_data, generated_data):
    """
    Report the performance of Post-hoc RNN one-step ahead prediction.
    (Irregular version: input & output dimension both = dim)

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    # Same steps as original code
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)

    hidden_dim = int(dim / 2)
    iterations = 2
    if seq_len > 96:
        if dim > 10:
            batch_size = 4
        else:
            batch_size = 16
        print('starting predictive score')
    else:
        batch_size = 128

    # Initialize the predictor model
    predictor_model = GRUPredictorIrregular(input_dim=dim, hidden_dim=hidden_dim)
    p_solver = optim.Adam(predictor_model.parameters())

    # Convert entire generated_data to torch Tensors for convenience
    # (We preserve the same list-of-arrays indexing approach)
    generated_data_torch = []
    for i in range(len(generated_data)):
        generated_data_torch.append(torch.tensor(generated_data[i], dtype=torch.float))

    # Training using Synthetic dataset
    for itt in range(iterations):
        idx = np.random.permutation(len(generated_data_torch))
        train_idx = idx[:batch_size]

        # Prepare mini-batch X_mb, Y_mb, T_mb same logic as TF code
        X_mb = [generated_data_torch[i][:-1] for i in train_idx]
        Y_mb = [generated_data_torch[i][1:] for i in train_idx]
        T_mb = [generated_time[i] - 1 for i in train_idx]

        # Stack them as padded batch
        #  (the code expects shape [batch, max_seq_len-1, dim])
        # We'll find the largest time length in this batch
        max_len_batch = max([x.shape[0] for x in X_mb])  # each x has shape (#timesteps, dim)
        # Pad them
        X_tensor = []
        Y_tensor = []
        for x, y in zip(X_mb, Y_mb):
            x_pad = F.pad(x, (0, 0, 0, max_len_batch - x.shape[0]))  # pad time dimension
            y_pad = F.pad(y, (0, 0, 0, max_len_batch - y.shape[0]))
            X_tensor.append(x_pad.unsqueeze(0))
            Y_tensor.append(y_pad.unsqueeze(0))
        X_tensor = torch.cat(X_tensor, dim=0)
        Y_tensor = torch.cat(Y_tensor, dim=0)

        # Forward pass
        y_hat = predictor_model(X_tensor, T_mb)

        # We must replicate the absolute difference loss from TF
        # p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
        # => L1 loss in PyTorch
        p_loss = F.l1_loss(y_hat, Y_tensor)

        # Backprop
        p_solver.zero_grad()
        p_loss.backward()
        p_solver.step()

    # Test the trained model on the original data
    ori_data_torch = []
    for i in range(len(ori_data)):
        ori_data_torch.append(torch.tensor(ori_data[i], dtype=torch.float))
    idx = np.random.permutation(len(ori_data_torch))
    train_idx = idx[:no]

    X_mb_test = [ori_data_torch[i][:-1] for i in train_idx]
    Y_mb_test = [ori_data_torch[i][1:] for i in train_idx]
    T_mb_test = [ori_time[i] - 1 for i in train_idx]

    # Predict
    pred_Y_curr = []
    with torch.no_grad():
        for i in range(no):
            x_i = X_mb_test[i].unsqueeze(0)  # [1, #timesteps, dim]
            t_i = [T_mb_test[i]]  # a single-element list for that sample
            y_hat_i = predictor_model(x_i, t_i).squeeze(0)  # [#timesteps, dim]
            pred_Y_curr.append(y_hat_i.cpu().numpy())

    # Compute performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        # Compare pred_Y_curr[i] vs Y_mb_test[i]
        # pred_Y_curr[i] is shape (#timesteps, dim)
        # Y_mb_test[i] is shape (#timesteps, dim)
        MAE_temp += mean_absolute_error(Y_mb_test[i].numpy(), pred_Y_curr[i])
    predictive_score = MAE_temp / no

    return predictive_score
