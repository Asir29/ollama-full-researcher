#!/usr/bin/env python3
"""
Defines the configuration for an NNI experiment.

Fra, Vittorio,
Politecnico di Torino,
EDA Group,
Torino, Italy.

Kaiser, Jakob
Institute/University
Group
Place

Muller-Cleve, Simon F.,
Istituto Italiano di Tecnologia - IIT,
Event-driven perception in robotics - EDPR,
Genova, Italy.
"""


import argparse

from nni.experiment import *


search_space = {
    'nb_hidden': {'_type': 'quniform', '_value': [30, 63, 1]}, # use it as an int in the script
    'alpha_r': {'_type': 'quniform', '_value': [0, 1, 0.05]},
    'alpha_out': {'_type': 'quniform', '_value': [0, 1, 0.05]},
    'beta_r': {'_type': 'quniform', '_value': [0, 1, 0.05]},
    'beta_out': {'_type': 'quniform', '_value': [0, 1, 0.05]},
    'lr': {'_type': 'choice', '_value': [0.0001, 0.00015, 0.0005, 0.001, 0.0015, 0.002, 0.005, 0.01, 0.1]},
    'reg_l1': {'_type': 'quniform', '_value': [0, 1e-3, 1e-4]},
    'reg_l2': {'_type': 'quniform', '_value': [0, 1e-5, 1e-6]},
    'slope': {'_type': 'quniform', '_value': [5, 20, 5]}, # use it as an int in the script
}
searchspace_filename = "train_snnTorch_NIR_Braille_searchspace"
searchspace_path = "./searchspaces/{}.json".format(searchspace_filename)
with open(searchspace_path, "w") as write_searchspace:
    json.dump(search_space, write_searchspace)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Name for the experiment
    parser.add_argument('-exp_name',
                        type=str,
                        default='NIR_braille_th1',
                        help='Name for the starting experiment.')

    # Maximum number of trials
    parser.add_argument('-exp_trials',
                        type=int,
                        default=10000,
                        help='Number of trials for the starting experiment.')

    # Maximum time
    parser.add_argument('-exp_time',
                        type=str,
                        default='100d',
                        help='Maximum duration of the starting experiment.')

    # How many (if any) GPUs are available
    parser.add_argument('-exp_gpu_number',
                        type=int,
                        default=2,
                        help='How many GPUs to use for the starting experiment.')

    # Which GPU to use
    parser.add_argument('-exp_gpu_sel',
                        type=int,
                        default=[0,1],
                        help='GPU index to be used for the experiment.')

    # How many trials at the same time
    parser.add_argument('-exp_concurrency',
                        type=int,
                        default=1,
                        help='Concurrency for the starting experiment.')
    
    # Max trials per GPU
    parser.add_argument('-max_per_gpu',
                        type=int,
                        default=5,
                        help='Maximum number of trials per GPU.')

    # What script to use for the experiment
    parser.add_argument('-script',
                        type=str,
                        default='main_nir.py',
                        help='Path of the training script.')

    # Which port to use
    parser.add_argument('-port',
                        type=int,
                        default=8081,
                        help='Port number for the starting experiment.')

    args = parser.parse_args()

    
    config = ExperimentConfig(
        experiment_name=args.exp_name,
        experiment_working_directory="~/nni-experiments/{}".format(
            args.exp_name),
        trial_command=f"python3 {args.script}",
        trial_code_directory="./",
        search_space=search_space,
        tuner=AlgorithmConfig(name="Anneal",
                              class_args={"optimize_mode": "maximize"}),
        assessor=AlgorithmConfig(name="Medianstop",
                                 class_args=({'optimize_mode': 'maximize',
                                              'start_step': 10})),
        tuner_gpu_indices=args.exp_gpu_sel,
        max_trial_number=args.exp_trials,
        max_experiment_duration=args.exp_time,
        trial_concurrency=args.exp_concurrency,
        training_service=LocalConfig(trial_gpu_number=args.exp_gpu_number,
                                     max_trial_number_per_gpu=args.max_per_gpu,
                                     use_active_gpu=True)
    )

    experiment = Experiment(config)

    experiment.run(args.port)

    # Stop through input
    input('Press any key to stop the experiment.')

    # Stop at the end
    experiment.stop()
