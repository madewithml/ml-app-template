import os
import json
import shutil
from threading import Thread

from document_classification.config import BASE_DIR
from document_classification.ml.utils import load_config, training_setup, training_operations


def train(config_filepath):
    """Asynchronously train a model.
    """

    # Load config and set up
    config = load_config(config_filepath=config_filepath)
    config = training_setup(config=config)

    # Asynchronous call
    thread = Thread(target=training_operations, args=(config,))
    thread.start()

    return config


def infer(config_filepath):
    """Predict using a model.
    """
    pass

def get_experiment_ids():
    """Get list of experiments.
    """
    # Experiments dir
    experiments_dir = os.path.join(BASE_DIR, "experiments")

    # Get experiements
    experiment_ids = [f for f in os.listdir(experiments_dir) \
        if os.path.isdir(os.path.join(experiments_dir, f))]

    # Only show valid experiments
    valid_experiment_ids = []
    for experiment_id in experiment_ids:
        experiment_details = get_experiment_info(experiment_id)
        if experiment_details["done_training"]:
            valid_experiment_ids.append(experiment_id)

    # Sort
    valid_experiment_ids = sorted(valid_experiment_ids)

    return valid_experiment_ids

def get_experiment_info(experiment_id):
    """ Get training info for the experiment.
    """

    # Get experiment details
    experiment_dir = os.path.join(BASE_DIR, "experiments", experiment_id)
    config_filepath = os.path.join(experiment_dir, "config.json")
    train_state_filepath = os.path.join(experiment_dir, "train_state.json")

    # Load files
    with open(config_filepath, 'r') as fp:
        config = json.load(fp)
    with open(train_state_filepath, 'r') as fp:
        train_state = json.load(fp)

    # Join info
    experiment_info = {**config, **train_state}

    return experiment_info

def delete_experiment(experiment_id):
    """Delete an experiment.
    """
    # Delete the experiment dir
    experiment_dir = os.path.join(BASE_DIR, "experiments", experiment_id)
    shutil.rmtree(experiment_dir)
    response = "Successfully deleted experiment id: {0}".format(experiment_id)
    return response

