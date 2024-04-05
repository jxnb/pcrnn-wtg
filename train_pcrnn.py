import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.linear_model import LinearRegression

from pcrnn.models.pcrnn import RNNCell, PCRNN, train
from pcrnn.models.baselines import BaselineRNN, NaiveForecaster
from pcrnn.evaluation import evaluate_model
from pcrnn.data_utils import *
from pcrnn.callbacks import EarlyStopping
from pcrnn.preprocessing import MinMaxScaler
from pcrnn.plotting import add_prediction_subplot, plot_losses


with open(Path('data', 'wtg_plants_turbines.json'), 'r') as f:  # set path to dataset dictionary
    wtg_dataset_dict = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = Path('/home/jxnb/data/wtg_data/paper_dataset') #Path('path/to/data')  # set path to data
n_epochs = 3# 100
batch_size = 64

rnn_input_size = 4
rnn_hidden_size = 16
rnn_num_layers = 1

pcrnn_R = 0.015
pcrnn_lambda1 = 0.02
pcrnn_lambda2 = 0.02

lags = 5
train_plant = "PlantA"
test_plants = ["PlantB", "PlantC"]
num_turbine_samples_train = 5
num_turbine_samples_test = 5

data_start = "2022-01"
data_split = "2023-01"

scaler = MinMaxScaler(feature_range=(0, 1), interval=(263.15, 393.15))


# === Import training data ===

train_datasets = []
rng = np.random.default_rng()
train_wtgs = [i for i in sample_wtgs(train_plant, num_turbine_samples_train, data_dict=wtg_dataset_dict, rng=rng)]

time_interval = set()
for _, train_wtg_turbine in train_wtgs:
    df = import_wtg_turbine_dataset(Path(data_dir, train_plant), train_wtg_turbine)
    df = df[df['date'] >= data_start]
    df_timedelta = get_time_deltas(df, unit='s')
    if len(df_timedelta) != 1:
        raise AssertionError(f"Data has different intervals: found {df_timedelta} seconds")
    time_interval.add(list(df_timedelta)[0])
    df['Gearbox Bearing Temp [K]'] = scaler.transform(df['Gearbox Bearing Temp [K]'])
    train_datasets.append((train_wtg_turbine, df))

if len(time_interval) != 1:
    raise AssertionError(f"Data has different intervals: found {time_interval} seconds")
time_interval = list(df_timedelta)[0]

data_train_list = []
data_test_list = []
for turbine_id, df in train_datasets:
    df_train, df_test = create_train_test_sets(df, percent_train=None, split_date=data_split, additional_lags=lags)
    data_train_list.append(df_train.to_numpy())
    data_test_list.append((df_test['date'], df_test.drop(columns='date').to_numpy()))
    
data_train = np.concatenate(data_train_list)
train_subset, val_subset = create_torch_train_set(data_train,
                                                val_set_ratio=0.2, 
                                                random_split=True,
                                                reshape_for_rnn=True,
                                                device=device,
                                                generator=None)                    

train_loader = DataLoader(dataset=train_subset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            drop_last=True,
                            generator=None)
val_loader = DataLoader(dataset=val_subset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        drop_last=True,
                        generator=None)


# === Train models ===

trained_models = []
model_names = []

# --- PC-RNN ---
rnn = RNNCell(rnn_input_size, rnn_hidden_size, rnn_num_layers)
pcrnn = PCRNN(rnn, pcrnn_R, pcrnn_lambda1, pcrnn_lambda2)
pcrnn.to(device)

criterion = nn.MSELoss()
early_stopping = EarlyStopping(delta=0, patience=10, restore=True, delete_on_restore=True)
    
optimizer = Adam(pcrnn.parameters())             
pcrnn, pcrnn_train_loss, pcrnn_best_epoch = train(pcrnn, optimizer, criterion, n_epochs, train_loader, val_loader, 
                    early_stopping=early_stopping,
                    device=device)

trained_models.append(pcrnn)
model_names.append("PCRNN")

# --- Baseline RNN ---
rnn = RNNCell(rnn_input_size, rnn_hidden_size, rnn_num_layers)
bnn = BaselineRNN(rnn)
bnn.to(device)

criterion = nn.MSELoss()
early_stopping = EarlyStopping(delta=0, patience=10, restore=True, delete_on_restore=True)
    
optimizer = Adam(bnn.parameters())             
bnn, bnn_train_loss, bnn_best_epoch = train(bnn, optimizer, criterion, n_epochs, train_loader, val_loader, 
                    early_stopping=early_stopping,
                    device=device)

trained_models.append(bnn)
model_names.append("Baseline RNN")

# --- Physics-based linear model ---
X_train_linreg, y_train_linreg = torch_dataset_to_numpy(train_subset)
X_train_linreg = create_linreg_dataset(X_train_linreg, use_lags=True)
linreg_model = LinearRegression()
linreg_model = linreg_model.fit(X_train_linreg, y_train_linreg)

trained_models.append(linreg_model)
model_names.append("Linear")

# --- Naive forecaster ---
naive_forecaster = NaiveForecaster()
trained_models.append(naive_forecaster)
model_names.append("Naive")

plot_losses(['PCRNN', 'Baseline RNN'], [pcrnn_train_loss, bnn_train_loss], [pcrnn_best_epoch, bnn_best_epoch])


# === Test split data evaluation ===

print('--- Test set results ---')
for (test_plant_name, test_turbine_name), (test_date, data_test) in zip(train_wtgs, data_test_list):
    print(test_plant_name, test_turbine_name)
    days = test_date.dt.date.value_counts()
    n_indices_per_day = int(3600 * 24 / time_interval)
    whole_days = days.index[days == n_indices_per_day]
    if whole_days.size != 0:
        plot_day = rng.choice(whole_days)
    else:
        plot_day = None
    
    fig, ax = plt.subplots(1, len(trained_models), figsize=(17, 5))
    rmses = []
    for i, (model_name, model) in enumerate(zip(model_names, trained_models)):
        X_test, y_test = split_data_and_targets(data_test)
        y_prev = X_test[:, -1, -1]
        if scaler is not None:
            y_test = scaler.inverse_transform(y_test)
            y_prev = scaler.inverse_transform(y_prev)
        true_grad = y_test - y_prev
        preds, squared_errors, _ = evaluate_model(model, X_test, y_test, true_grad, device, scaler)
        rmse = np.sqrt(np.mean(squared_errors))
        rmses.append(rmse)
        data_freq = f"{int(time_interval / 60)}min"
        add_prediction_subplot(test_date, 
                               preds, 
                               y_test, 
                               ax=ax[i], 
                               data_freq=data_freq, 
                               plot_date=plot_day, 
                               ylabel=f"Bearing temp [K]", 
                               title=f'{model_name}')
    s = ", ".join([f"{mname}: {rmse:.4f}" for mname, rmse in zip(model_names, rmses)])
    print(f'RMSEs: {s}')
    fig.suptitle(f"{test_plant_name}, {test_turbine_name}")
    fig.tight_layout()
    plt.show()


# === Generalization evaluation (unseen plants/turbines)===

print('--- Generalization set results ---')
for test_plant in test_plants:
    test_turbine_list = [dev for dev in wtg_dataset_dict[test_plant] if dev not in [w[1] for w in train_wtgs]]
    test_turbines = rng.choice(test_turbine_list, size=num_turbine_samples_test, replace=False)

    for wtg_turbine in test_turbines:
        print(test_plant, wtg_turbine)
        df = import_wtg_turbine_dataset(Path(data_dir, test_plant), wtg_turbine)
        df = df[df['date'] >= data_split]
        df_timedelta = get_time_deltas(df, unit='s')
        if len(df_timedelta) != 1:
            raise AssertionError(f"Data has different intervals: found {df_timedelta} seconds")
        if list(df_timedelta)[0] != time_interval:
            raise AssertionError(f"Data has different time interval than training data ({df_timedelta[0]} s, expected {time_interval} s)")
        df['Gearbox Bearing Temp [K]'] = scaler.transform(df['Gearbox Bearing Temp [K]'])
        df_test = create_train_test_sets(df, additional_lags=lags)
        test_date = df_test['date']
        data_test = df_test.drop(columns='date').to_numpy()

        days = test_date.dt.date.value_counts()
        n_indices_per_day = int(3600 * 24 / time_interval)
        whole_days = days.index[days == n_indices_per_day]
        if whole_days.size != 0:
            plot_day = rng.choice(whole_days)
        else:
            plot_day = None
        fig, ax = plt.subplots(1, len(trained_models), figsize=(17, 5))

        rmses = []
        for i, (model_name, model) in enumerate(zip(model_names, trained_models)):
            X_test, y_test = split_data_and_targets(data_test)
            y_prev = X_test[:, -1, -1]                            
            if scaler is not None:
                y_test = scaler.inverse_transform(y_test)
                y_prev = scaler.inverse_transform(y_prev)
            true_grad = y_test - y_prev

            preds, squared_errors, _ = evaluate_model(model, X_test, y_test, true_grad, device, scaler)
            rmse = np.sqrt(np.mean(squared_errors))
            rmses.append(rmse)
            data_freq = f"{int(time_interval / 60)}min"
            add_prediction_subplot(test_date, 
                                   preds, 
                                   y_test, 
                                   ax=ax[i], 
                                   data_freq=data_freq,
                                   plot_date=plot_day,
                                   ylabel=f"Bearing temp [K]", 
                                   title=f'{model_name}')
            
        s = ", ".join([f"{mname}: {rmse:.4f}" for mname, rmse in zip(model_names, rmses)])
        print(f'RMSEs: {s}')
        fig.suptitle(f"{test_plant}, {test_turbine_name}")
        fig.tight_layout()
        plt.show()


