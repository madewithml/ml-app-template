import os
import json
import torch
from threading import Thread

from document_classification.utils import BASE_DIR
from document_classification.ml.utils import load_config, training_setup, training_operations
from document_classification.ml.load import load_data
from document_classification.ml.split import split_data
from document_classification.ml.preprocess import preprocess_text, preprocess_data
from document_classification.ml.vectorizer import Vectorizer
from document_classification.ml.dataset import Dataset, sample
from document_classification.ml.model import initialize_model
from document_classification.ml.training import Trainer
from document_classification.ml.inference import Inference

def train(config_filepath="/Users/goku/Documents/productionML/src/document_classification/configs/config.json"):
    """Train a model.
    """

    # Load config and set up
    config = load_config(config_filepath="/Users/goku/Documents/productionML/src/document_classification/configs/config.json")
    config = training_setup(config=config)

    # Asynchronous call
    thread = Thread(target=training_operations, args=(config,))
    thread.start()

    # results
    results = {
        "experiment_id": config["experiment_id"],
    }

    return results


def infer(config_filepath):
    """Predict using a model.
    """
    pass

if __name__ == '__main__':
    train()
