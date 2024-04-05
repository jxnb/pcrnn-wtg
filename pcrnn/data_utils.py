import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, Subset


def get_all_wtgs(plant, data_dict):
    return [(plant, device) for device in data_dict[plant]]


def sample_wtgs(plant, n_samples, data_dict, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    wtgs = [(plant, device) for device in data_dict[plant]]
    sample_idx = rng.choice(len(wtgs), size=n_samples, replace=False)
    samples = [wtgs[i] for i in sample_idx]
    return samples


def import_wtg_turbine_dataset(dir, wtg_device):
    if not wtg_device.endswith(".csv"):
        df = pd.read_csv(Path(dir, f"{wtg_device}.csv"))
    else:
        df = pd.read_csv(Path(dir, f"{wtg_device}"))
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_time_deltas(dataframe, unit='s'):
    timedeltas = set()
    for tdelta in np.unique(dataframe['date'][1:].values - dataframe['date'][:-1].values) / np.timedelta64(1, unit):
        timedeltas.add(tdelta)
    return timedeltas


def create_train_test_sets(df, percent_train=None, split_date=None, additional_lags=5, exclude_downtimes=True):
    """Creates train/test or only test sets from wind turbine dataframes.

    Parameters
    ----------
    df
        Pandas dataframe.
    percent_train
        Percentage of data to be used as training set.
    split_date
        Date to split training and test sets. If neither percent_train or split_date are given,
        the function returns a single dataset.
    additional_lags
        Additional lags or lookbacks included in a single state.
    exclude_downtimes
        If True, turbine downtimes (Status == 0) are excluded from the data.
    """
    
    date = df['date']
    if exclude_downtimes:
        df.loc[df['Status'] == 0] = np.nan
    df = df.drop(columns=['date', 'Status'])

    dataset = pd.merge(df.shift(1), df, left_index=True, right_index=True, suffixes=(' (t-1)', None))
    dataset_columns = dataset.columns

    if additional_lags > 0:
        for i in range(additional_lags):
            dataset = pd.merge(df.shift(i+2), dataset, left_index=True, right_index=True, suffixes=(f' (t-{i+2})', None))
        lag_timesteps = [f'(t-{i})' for i in range(1, additional_lags + 2)]
        lag_timesteps.reverse()
        dataset_columns = [*[f"{col} {i}" for i in lag_timesteps[:-1] for col in df.columns], *dataset_columns]

    dataset['date'] = date
    dataset = dataset.dropna()
    if percent_train:
        split_idx = (len(dataset) * percent_train) // 100
        data_train = dataset.iloc[:split_idx]
        data_test = dataset.iloc[split_idx + additional_lags + 2:]
        data_train = data_train[dataset_columns]
        data_test = data_test[['date', *dataset_columns]]
        return data_train, data_test
    elif split_date:
        split_idx = len(dataset[dataset['date'] < split_date])
        data_train = dataset.iloc[:split_idx - (additional_lags + 2)]
        data_test = dataset.iloc[split_idx:]
        data_train = data_train[dataset_columns]
        data_test = data_test[['date', *dataset_columns]]
        return data_train, data_test
    else:
        data_test = dataset[['date', *dataset_columns]]    
        return data_test


def split_data_and_targets(data_array, rnn=True):
    """Splits inputs and targets and reshapes data for RNN"""
    input_data = data_array[:, :-1]
    targets = data_array[:, -1]

    xs = []
    for x in input_data:
        seq_length = int((len(x) + 1) / 4)
        x_sub = x[:-3].reshape(seq_length - 1, 4)
        prev_vals = x_sub[:, -1:]
        new_row = x[-3:][None,...]
        current_vals = np.concatenate((x_sub[:, :-1], new_row), axis=0)
        new_x = np.concatenate([current_vals[1:,:], prev_vals], axis=1)
        xs.append(new_x)

    if rnn:
        X = np.concatenate([x[None,...] for x in xs], axis=0)
    else:
        X = np.concatenate([x.flatten()[None, ...] for x in xs], axis=0)
    return X, targets


def create_torch_train_set(data, val_set_ratio=0.2, random_split=True, reshape_for_rnn=True, generator=None, device=None):
    """Create torch dataset with train/validation split from numpy data array.

    If reshape_for_rnn is True, a single input state is a matrix of shape (n+1) x 4, where n is the 
    number of additional lags given to create_train_test_sets:
    
    x = [
          power_t-n  temp_ambient_t-n  rotor_speed_t-n  temp_bearing_t-n-1
          .
          .
          .
          power_t-1  temp_ambient_t-1  rotor_speed_t-1  temp_bearing_t-2
          power_t    temp_ambient_t    rotor_speed_t    temp_bearing_t-1
        ]
    

    Parameters
    ----------
    data
        Data array create with create_train_test_sets().
    val_set_ratio
        Part of data to be used as validation set.
    random_split
        Whether validation set should be random data from the training set. If False, the last x data
        points are cut off for the validation set.
    reshape_for_rnn
        If True, data is reshaped including lags to be used as input for the RNN models.
    generator
        Torch random number generator.
    device
        Torch device.
    """

    X, y = split_data_and_targets(data, rnn=reshape_for_rnn)
    
    dataset_train = TensorDataset(torch.tensor(X).float().to(device), 
                                    torch.tensor(y).unsqueeze(1).float().to(device))
    
    if val_set_ratio > 0:
        train_set_size = int(np.ceil(X.shape[0])*(1 - val_set_ratio))
        val_set_size = int(X.shape[0] - train_set_size)
        if random_split:
            train_subset, val_subset = torch.utils.data.random_split(dataset_train, 
                                                                    [train_set_size, val_set_size], 
                                                                    generator=generator)
        else:
            indices = np.arange(data.shape[0])
            train_subset = Subset(dataset_train, indices[:train_set_size])
            val_subset = Subset(dataset_train, indices[train_set_size:])
        return train_subset, val_subset
    else:
        return dataset_train


def torch_dataset_to_numpy(dataset):
    """Converts a torch tensor dataset to a numpy array."""

    x_list = []
    y_list = []
    for row in list(dataset):
        x = row[0]
        y = row[1]
        x_list.append(x.cpu().numpy()[None,...])
        y_list.append(y.cpu().numpy())
    X = np.concatenate(x_list,axis=0)
    y = np.concatenate(y_list,axis=0)
    return X, y


def create_linreg_dataset(data, use_lags=True):
    """Reshape RNN dataset for baseline linear physics model."""
    if use_lags:
        dataset = data.reshape(data.shape[0], -1)
    else:
        dataset = data[:, -1, :]
    return dataset
