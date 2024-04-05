import numpy as np
from pathlib import Path
from timeit import default_timer as timer
import json

from run_experiments import run_experiment


"""Script to reproduce results from paper"""


data_dir = Path('/path/to/dataset')
test_dir_name = "results"

paper_results_dir = Path('paper_results')

with open('data/wtg_plants_turbines.json', 'r') as f:
    wtg_data = json.load(f)

wtg_plants = list(wtg_data.keys())
test_plants = list(wtg_data.keys())

print(wtg_plants)
print(test_plants)

exp_dirs = sorted(list(p[0] for p in paper_results_dir.walk() if p[1] == []))

for exp_dir in exp_dirs:
    with open(Path(exp_dir, 'experiment_parameters.json'), 'r') as f:
        experiment_parameters = json.load(f)
    for m, v in experiment_parameters['models'].items():
        v['class'] = eval(v['class'])

    train_plant = experiment_parameters['dataset_parameters']['train_plants']
    n_train_turbines = experiment_parameters['dataset_parameters']['n_train_samples']

    test_id = f"{train_plant}_{n_train_turbines}"            
    result_dir = Path(test_dir_name, train_plant, test_id)
    result_dir.mkdir(parents=True, exist_ok=True)

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
