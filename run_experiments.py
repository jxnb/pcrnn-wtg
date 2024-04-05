import numpy as np
import pandas as pd
from pathlib import Path
import random
from timeit import default_timer as timer
import json
from collections import defaultdict
from copy import deepcopy

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


def run_experiment(models,
                   rnn_parameters,
                   train_parameters,
                   wtg_data_dict, 
                   train_plant,
                   test_plants,
                   data_dir, 
                   out_dir,
                   num_train_samples=1,
                   num_test_samples=5,
                   train_sample_dict=None,
                   test_sample_dict=None,
                   num_iterations=1,
                   seed_file=None,
                   torch_device=None,
                   store_models=True,
                   disable_output=False):
    """Run experiments. Pass seed file to reproduce results from paper.

    Parameters
    ----------
    models : dict
        Dictionary with models to train of the following structure:

        {
            model_name: {
                "class": model_class,
                "params": {}
            },
            .
            .
            .
        }

        where model_class can be any class (not instance!) from pcrnn.models. The "params" key contains
        class initialization arguments as dict (if any, otherwise empty dict).
        Example: 

        {
            "PCRNN": {
                "class": PCRNN,
                "params": {
                    "R": 0.015,
                    "lambda1": 0.02,
                    "lambda2": 0.02
                }
            },
            "RNN": {
                "class": BaselineRNN,
                "params": {}
            },
            "Linear": {
                "class": LinearRegression,
                "params": {}
            },
            "Naive": {
                "class": NaiveForecaster,
                "params": {}
            }
        }
    rnn_parameters : dict
        Dictionary with RNNCell arguments.
    wtg_data_dict : dict
        Dictionary with WTG datasets: plants as keys and list of turbine ids as values.
        (See wtg_plants_turbines.json)
    train_plant : str
        Name of WTG dataset used for training.
    test_plants : list
        List of WTG dataset names used for generalization tests.
    data_dir : str
        Path to dataset
    out_dir : str, pathlib.PosixPath
        Path to output directory
    num_train_samples : int, optional
        Number of WTG turbines used for training from train_plant, default 1
    num_test_samples : int, optional
        Number of WTG turbines used for generalization testing per plant in test_plants, default 5
    train_sample_dict : dict, optional
        Dictionary with WTG datasets with plants as keys and list of turbine ids as values.
        Can be used to specify which turbines the model is be trained with, default None
    test_sample_dict : dict, optional
        Dictionary with WTG datasets with plants as keys and list of turbine ids as values.
        Can be used to specify which turbines the model is be tested with, default None
    num_iterations : int, optional
        Number of independent experiment runs, default 1
    seed_file : Path to file where random seeds are stored to reproduce experiment runs, default None.
    torch_device : str, optional
        Torch device, either 'cpu' or 'cuda'. If None, CUDA is used if available, default None
    store_models : bool, optional
        If True, trained models for every iteration are saved, default True
    disable_output : bool, optional
        If True, training runs with reduced verbosity (no progress bars), default False
    """
    
    if torch_device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(torch_device)

    test_id = out_dir.stem

    if seed_file:
        with open(seed_file, 'r') as f:
            seeds = json.load(f)
            initial_seed = seeds['main_seed']
            train_seeds = seeds['train_seeds']
            dataset_seeds = seeds['dataset_seeds']
            num_iterations = len(train_seeds)
    else:
        initial_seed = torch.Generator().seed()
    rng = np.random.default_rng(seed=initial_seed)

    n_epochs = int(train_parameters["epochs"])
    batch_size = int(train_parameters["batch_size"])

    # track experiment
    experiment_log = {}
    for i in range(num_iterations):
        experiment_log[i+1] = defaultdict(lambda: defaultdict(dict))
    
    # track test set MSEs
    test_set_mses = []

    # track generalization MSEs
    generalization_mses = []

    # track losses
    model_losses = defaultdict(dict)

    # track random seeds
    seed_log = {}
    seed_log['main_seed'] = initial_seed
    seed_log['train_seeds'] = []
    seed_log['dataset_seeds'] = []

    # Run every combination x times
    for it in range(num_iterations):
        iteration = it+1
        print(f"\nTest iteration {iteration}/{num_iterations}\n")

        scaler = MinMaxScaler(feature_range=(0, 1), interval=(263.15, 393.15))

        iter_log = experiment_log[it+1]

        train_datasets = []
        if train_sample_dict is None:
            if (num_train_samples is not None) and (num_train_samples < len(wtg_data_dict[train_plant])):
                train_wtgs = [i for i in sample_wtgs(train_plant, num_train_samples, wtg_data_dict, rng=rng)]
            else:
                train_wtgs = get_all_wtgs(train_plant, wtg_data_dict)
        else:
            train_wtgs = [(k, v) for k in train_sample_dict.keys() for v in train_sample_dict[k]]

        time_interval = set()
        for _, train_wtg_turbine in train_wtgs:
            df = import_wtg_turbine_dataset(Path(data_dir, train_plant), train_wtg_turbine)
            df = df[df['date'] >= train_parameters["start_date"]]
            df_timedelta = get_time_deltas(df, unit='s')
            if len(df_timedelta) != 1:
                raise AssertionError(f"Data has different intervals: found {df_timedelta} seconds")
            time_interval.add(list(df_timedelta)[0])
            df['Gearbox Bearing Temp [K]'] = scaler.transform(df['Gearbox Bearing Temp [K]'])
            train_datasets.append((train_wtg_turbine, df))

        if len(time_interval) != 1:
            raise AssertionError(f"Data has different intervals: found {time_interval} seconds")
        time_interval = list(df_timedelta)[0]

        iter_log['train_turbines'] = [w[1] for w in train_wtgs]
        iter_log['n_train_turbines'] = len(train_wtgs)
        iter_log['test_wtgs'] = {}

        data_train_list = []
        data_test_list = []

        for turbine_id, df in train_datasets:
            df_train, df_test = create_train_test_sets(df, 
                                                        percent_train=None, 
                                                        split_date=train_parameters["split_date"],
                                                        additional_lags=train_parameters["lags"])
            data_train_list.append(df_train.to_numpy())
            data_test_list.append((df_test['date'], df_test.drop(columns='date').to_numpy()))
            
        data_train = np.concatenate(data_train_list)

        trained_models = {}
        
        if seed_file:
            dataset_seed = dataset_seeds[it]
            train_seed = train_seeds[it]
        else:
            dataset_seed = torch.Generator().seed()
            train_seed = torch.Generator().seed()

        seed_log['dataset_seeds'].append(dataset_seed)
        seed_log['train_seeds'].append(train_seed)

        for model_name, model_dict in models.items():
            model_class = model_dict['class']
            model_params = model_dict['params']

            print(f"Train {model_name}...")

            dataset_rng = torch.Generator().manual_seed(dataset_seed)

            rnn_reshape = True if issubclass(model_class, torch.nn.Module) else False
            torch_dev = device if issubclass(model_class, torch.nn.Module) else None
            train_subset, val_subset = create_torch_train_set(data_train,
                                                            val_set_ratio=0.2, 
                                                            random_split=True,
                                                            reshape_for_rnn=rnn_reshape,
                                                            device=torch_dev, 
                                                            generator=dataset_rng)                    
            torch.manual_seed(train_seed)
            random.seed(train_seed)
            
            # train Neural Networks
            if issubclass(model_class, torch.nn.Module):
                train_loader = DataLoader(dataset=train_subset, 
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          drop_last=True,
                                          generator=dataset_rng)
                val_loader = DataLoader(dataset=val_subset, 
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        drop_last=True,
                                        generator=dataset_rng)
                rnn = RNNCell(**rnn_parameters)
                net = model_class(rnn, **model_params)
                net.to(device)

                criterion = nn.MSELoss()
                early_stopping = EarlyStopping(delta=train_parameters["early_stopping_delta"], 
                                                patience=train_parameters["early_stopping_patience"], 
                                                model_path=str(Path(out_dir, 
                                                                    "trained_models", 
                                                                    f"{model_name}_{str(iteration).zfill(2)}.pt")),
                                                restore=True,
                                                delete_on_restore=not store_models)
                
                optimizer = Adam(net.parameters())             
                net_trained, net_loss_dict, net_best_epoch = train(net, 
                                                                   optimizer, 
                                                                   criterion, 
                                                                   n_epochs, 
                                                                   train_loader, 
                                                                   val_loader, 
                                                                   early_stopping=early_stopping,
                                                                   device=device,
                                                                   disable_output=disable_output)
                
                trained_models[model_name] = net_trained
                model_losses[model_name][iteration] = net_loss_dict

            elif model_class == LinearRegression:
                X_train_linreg, y_train_linreg = torch_dataset_to_numpy(train_subset)
                X_train_linreg = create_linreg_dataset(X_train_linreg, use_lags=True)
                linreg_model = LinearRegression()
                linreg_model = linreg_model.fit(X_train_linreg, y_train_linreg)
                trained_models[model_name] = linreg_model
                
            elif model_class == NaiveForecaster:
                naive_forecaster = NaiveForecaster()
                trained_models[model_name] = naive_forecaster


        # --- Test data evaluation ---
       
        print(f"Evaluate models on test data...")

        for (_, test_turbine_name), (test_date, data_test) in zip(train_wtgs, data_test_list):
            for model_name, model in trained_models.items():
                model_params = models[model_name]
                X_test, y_test = split_data_and_targets(data_test)
                y_prev = X_test[:, -1, -1]
                if scaler is not None:
                    y_test = scaler.inverse_transform(y_test)
                    y_prev = scaler.inverse_transform(y_prev)
                true_grad = y_test - y_prev
                _, pred_errors, _ = evaluate_model(model, X_test, y_test, true_grad, device, scaler)

                mse = np.mean(pred_errors)
                n_evals = len(pred_errors)
                test_set_mses.append({'iteration': iteration, 
                                      'model_name': model_name, 
                                      'mse': mse, 
                                      'eval_count': n_evals})


        # --- Generalization evaluation ---

        print(f"Evaluate models on unseen data...")

        if test_sample_dict:
            test_plants = list(test_sample_dict.keys())
        for test_plant in test_plants:
            gen_errors = np.zeros(len(trained_models.keys()))
            gen_nevals = np.zeros_like(gen_errors)
            if test_sample_dict:
                test_turbine_list = [(k, v) for k in test_sample_dict.keys() for v in test_sample_dict[k]]
            else:
                test_turbine_list = [dev for dev in wtg_data_dict[test_plant] if dev not in [w[1] for w in train_wtgs]]
            if (num_test_samples is not None) and (num_test_samples < len(test_turbine_list)):
                test_turbines = rng.choice(test_turbine_list, size=num_test_samples, replace=False)
            else:
                test_turbines = test_turbine_list

            iter_log['test_wtgs'][test_plant] = test_turbines

            for wtg_turbine in test_turbines:
                df = import_wtg_turbine_dataset(Path(data_dir, test_plant), wtg_turbine)
                df = df[df['date'] >= train_parameters["split_date"]]
                df_timedelta = get_time_deltas(df, unit='s')
                if len(df_timedelta) != 1:
                    raise AssertionError(f"Data has different intervals: found {df_timedelta} seconds")
                if list(df_timedelta)[0] != time_interval:
                    raise AssertionError(f"Data has different time interval than training data ({df_timedelta[0]} s, expected {time_interval} s)")
                df['Gearbox Bearing Temp [K]'] = scaler.transform(df['Gearbox Bearing Temp [K]'])
                df_test = create_train_test_sets(df, additional_lags=train_parameters["lags"])
                test_date = df_test['date']
                data_test = df_test.drop(columns='date').to_numpy()
                
                for midx, (model_name, model) in enumerate(trained_models.items()):
                    model_params = models[model_name]
                    X_test, y_test = split_data_and_targets(data_test)
                    y_prev = X_test[:, -1, -1]                            
                    if scaler is not None:
                        y_test = scaler.inverse_transform(y_test)
                        y_prev = scaler.inverse_transform(y_prev)
                    true_grad = y_test - y_prev

                    _, pred_errors, _ = evaluate_model(model, X_test, y_test, true_grad, device, scaler)
                    
                    gen_errors[midx] += np.sum(pred_errors)
                    gen_nevals[midx] += n_evals

            for midx, model_name in enumerate(trained_models.keys()):
                generalization_mses.append({'iteration': iteration,
                                            'test_plant': test_plant, 
                                            'model_name': model_name, 
                                            'mse': gen_errors[midx] / gen_nevals[midx], 
                                            'eval_count': gen_nevals[midx]})

    with open(Path(out_dir, 'experiment_log.json'), 'w') as f:
        json.dump(experiment_log, f, indent=4)

    with open(Path(out_dir, 'seeds.json'), 'w') as f:
        json.dump(seed_log, f, indent=4)

    with open(Path(out_dir, f'model_losses.json'), 'w') as f:
        json.dump(model_losses, f, indent=4)

    test_df = pd.DataFrame.from_records(test_set_mses)
    test_df.to_csv(Path(out_dir, 'results_test_set.csv'), index=False)

    gen_df = pd.DataFrame.from_records(generalization_mses)
    gen_df.to_csv(Path(out_dir, 'results_generalization.csv'), index=False)


if __name__ == "__main__":
    
    data_dir = Path('/path/to/dataset')
    test_dir_name = "results"

    with open('data/wtg_plants_turbines.json', 'r') as f:
        wtg_data = json.load(f)

    wtg_plants = list(wtg_data.keys())
    test_plants = list(wtg_data.keys())

    print(wtg_plants)
    print(test_plants)

    number_of_train_turbines = [1, 3, 6, 9]

    for train_plant in wtg_plants:
        for n_train_turbines in number_of_train_turbines:

            experiment_parameters = dict(
                models = {
                    "PCRNN": {
                        "class": PCRNN,
                        "params": {
                            "R": 0.015,
                            "lambda1": 0.02,
                            "lambda2": 0.02
                        }
                    },
                    "RNN": {
                        "class": BaselineRNN,
                        "params": {}
                    },
                    "Linear": {
                        "class": LinearRegression,
                        "params": {}
                    },
                    "Naive": {
                        "class": NaiveForecaster,
                        "params": {}
                    }
                },
                rnn_parameters = {
                    "input_dims": 4,
                    "hidden_dims": 16,
                    "num_rnn_layers": 1
                },
                train_parameters = {
                    "epochs": 150,
                    "lags": 5,
                    "batch_size": 16,
                    "early_stopping_patience": 10,
                    "early_stopping_delta": 0,
                    "start_date": "2022-01",
                    "split_date": "2023-01"
                },
                dataset_parameters = {
                    "train_plants": train_plant,
                    "test_plants": test_plants,
                    "n_train_samples": n_train_turbines,
                    "n_test_samples": None
                }
            )

            test_id = f"{train_plant}_{n_train_turbines}"            
            result_dir = Path(test_dir_name, train_plant, test_id)
            result_dir.mkdir(parents=True, exist_ok=True)

            experiment_serializable_dict = deepcopy(experiment_parameters)
            for m, v in experiment_serializable_dict['models'].items():
                v['class'] = v['class'].__name__
            with open(Path(result_dir, 'experiment_parameters.json'), 'w') as f:
                json.dump(experiment_serializable_dict, f, indent=4)

            print(f'\n=== Starting test {test_id}: {train_plant}, {n_train_turbines} turbines ===\n')
            
            start = timer()
            run_experiment(models=experiment_parameters['models'],
                           rnn_parameters=experiment_parameters['rnn_parameters'],
                           train_parameters=experiment_parameters['train_parameters'],
                           wtg_data_dict=wtg_data,
                           train_plant=experiment_parameters["dataset_parameters"]['train_plants'],
                           test_plants=experiment_parameters["dataset_parameters"]['test_plants'],
                           data_dir=data_dir, 
                           out_dir=result_dir,
                           num_train_samples=experiment_parameters["dataset_parameters"]['n_train_samples'],
                           num_test_samples=experiment_parameters["dataset_parameters"]['n_test_samples'],
                           train_sample_dict=None,
                           test_sample_dict=None,
                           num_iterations=10,
                           seed_file=None,
                           torch_device='cpu',
                           store_models=True,
                           disable_output=False)

            end = timer()
            duration_min = np.round((end - start) / 60, decimals=2)
            duration_h = np.round((end - start) / 3600, decimals=2)
            print("Experiment done.")
            print(f"Total run time: {duration_min} min, {duration_h} hours")
