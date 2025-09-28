import numpy as np
import os
import sys
import torch
import torch.utils.data as Data
from numpy.f2py.auxfuncs import throw_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data

def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')

class MujocoDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, missing_rate=0.0):
        import pathlib
        here = pathlib.Path(__file__).resolve().parent.parent
        base_loc = here / 'data'
        loc = pathlib.Path(base_loc)
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.samples = tensors['mujoco_irregular']
            self.original_sample = tensors['mujoco_regular']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            # Apply missing rate to samples
            generator = torch.Generator().manual_seed(56789)
            for i in range(len(self.samples)):
                removed_points = torch.randperm(self.samples[i].shape[0], generator=generator)[
                                 :int(self.samples[i].shape[0] * missing_rate)].sort().values
                self.samples[i][removed_points, :-1] = float('nan')  # Set missing data to NaN except time column

            self.size = len(self.samples)
        else:
            raise FileNotFoundError


    def __getitem__(self, index):
        return self.original_sample[index], self.samples[index]

    def __len__(self):
        return len(self.samples)

def sine_data_generation(no, seq_len, dim, missing_rate):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    # data = list()
    irregular_dataset = list()
    ori_data = list()
    generator = torch.Generator().manual_seed(56789)
    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        ori_data.append(temp.copy())

        # Create irregular data
        removed_points = torch.randperm(temp.shape[0], generator=generator)[:int(temp.shape[0] * missing_rate)].sort().values
        temp[removed_points] = float('nan')
        idx = np.array(range(seq_len)).reshape(-1, 1)
        temp = np.concatenate((temp, idx), axis=1)
        irregular_dataset.append(temp)

    return ori_data, irregular_dataset


def add_gaussian_noise(data, noise_level=0.1):
    """
    Add Gaussian noise to the data.

    Args:
      - data: input data (numpy array)
      - noise_level: standard deviation of the Gaussian noise

    Returns:
      - noisy_data: data with added Gaussian noise
    """
    noise = np.random.normal(0, noise_level, data.shape)  # Mean=0, Std=noise_level
    noisy_data = data + noise
    return noisy_data


def real_data_loading(data_name, seq_len, missing_rate, gaussian_noise_level=0):
    """Load and preprocess real-world data.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    assert data_name in ['stock', 'energy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity']

    if data_name == 'stock':
        ori_data = np.loadtxt('./data/stock.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('./data/energy.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTh1':
        ori_data = np.loadtxt('./data/ETTh1.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTh2':
        ori_data = np.loadtxt('./data/ETTh2.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTm1':
        ori_data = np.loadtxt('./data/ETTm1.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTm2':
        ori_data = np.loadtxt('./data/ETTm2.csv', delimiter=",", skiprows=1)
    elif data_name == 'weather':
        ori_data = np.loadtxt('./data/weather.csv', delimiter=",", skiprows=1)
    elif data_name == 'electricity':
        ori_data = np.loadtxt('./data/electricity.csv', delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    irregular_dataset = ori_data.copy()
    if gaussian_noise_level > 0:
        irregular_dataset = add_gaussian_noise(ori_data, gaussian_noise_level)
    generator = torch.Generator().manual_seed(56789)

    removed_points = torch.randperm(ori_data.shape[0], generator=generator)[
                     :int(ori_data.shape[0] * missing_rate)].sort().values
    irregular_dataset[removed_points] = float('nan')
    total_length = len(ori_data)
    index = np.array(range(total_length)).reshape(-1, 1)

    # Preprocess the data
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the data (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    # Preprocess the data
    irregular_dataset = np.concatenate((irregular_dataset, index), axis=1)
    irregular_temp_data = []
    # Cut data by sequence length
    for i in range(0, len(irregular_dataset) - seq_len):
        _x = irregular_dataset[i:i + seq_len]
        irregular_temp_data.append(_x)

    # Mix the data (to make it similar to i.i.d)
    irregular_data = []
    for i in range(len(irregular_temp_data)):
        irregular_data.append(irregular_temp_data[idx[i]])

    return data, irregular_data


def gen_dataloader(args):
    if args.dataset == 'sine':
        args.dataset_size = 10000
        ori_data, irregular_data = sine_data_generation(args.dataset_size, args.seq_len, args.input_channels, args.missing_rate)
        ori_data = torch.Tensor(np.array(ori_data))
        ori_train_set = Data.TensorDataset(ori_data)
        irregular_data = torch.Tensor(np.array(irregular_data))
        irregular_train_set = Data.TensorDataset(irregular_data)

    elif args.dataset in ['stock', 'energy', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity']:
        ori_data, irregular_data_np = real_data_loading(args.dataset, args.seq_len, missing_rate=args.missing_rate, gaussian_noise_level=args.gaussian_noise_level)
        ori_data = torch.Tensor(np.array(ori_data))
        ori_train_set = Data.TensorDataset(ori_data)
        irregular_data = torch.Tensor(np.array(irregular_data_np))
        irregular_train_set = Data.TensorDataset(irregular_data)

    elif args.dataset in ['mujoco']:
        train_set = MujocoDataset(args.seq_len, args.dataset, missing_rate=args.missing_rate)
        ori_data = list()
        irregular_data = list()
        for ori_data_b, irregular_data_b in train_set:
            ori_data.append(ori_data_b)
            irregular_data.append(irregular_data_b)
        ori_data = torch.Tensor(np.array(ori_data))
        ori_train_set = Data.TensorDataset(ori_data)
        irregular_data = torch.Tensor(np.array(irregular_data))
        irregular_train_set = Data.TensorDataset(irregular_data)

    train_loader = Data.DataLoader(dataset=irregular_train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    test_loader = Data.DataLoader(dataset=ori_train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    # For the time series benchmark, the entire dataset for both training and testing
    return train_loader, test_loader, None