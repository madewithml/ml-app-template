import os
import json
import numpy as np
import random
import time
from threading import Thread
import torch
import uuid

from document_classification.utils import BASE_DIR, create_dirs
from document_classification.ml.load import load_data
from document_classification.ml.split import split_data
from document_classification.ml.preprocess import preprocess_text, preprocess_data
from document_classification.ml.vectorizer import Vectorizer
from document_classification.ml.dataset import Dataset, sample
from document_classification.ml.model import initialize_model
from document_classification.ml.training import Trainer
from document_classification.ml.inference import Inference

def load_config(config_filepath):
    """Load the yaml config.
    """
    with open(config_filepath, 'r') as fp:
        config = json.load(fp)
    return config


def set_seeds(seed, cuda):
    """ Set Numpy and PyTorch seeds.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def generate_unique_id():
    """Generate a unique uuid
    preceded by a epochtime.
    """
    timestamp = int(time.time())
    unique_id = "{}_{}".format(timestamp, uuid.uuid1())

    return unique_id


def training_setup(config):
    """Set up training config.
    """
    # Set seeds
    set_seeds(seed=config["seed"], cuda=config["cuda"])

    # Generate experiment ID
    config["experiment_id"] = generate_unique_id()

    # Expand file paths
    config["save_dir"] = os.path.join(
        BASE_DIR, config["save_dir"], config["experiment_id"])
    create_dirs(config["save_dir"])
    config["vectorizer_file"] = os.path.join(
        config["save_dir"], config["vectorizer_file"])
    config["model_file"] = os.path.join(
        config["save_dir"], config["model_file"])

    # Save config
    config_fp = os.path.join(config["save_dir"], "config.json")
    with open(config_fp, "w") as fp:
        json.dump(config, fp)

    # Check CUDA
    if not torch.cuda.is_available():
        config["device"] = False
    config["device"] = torch.device("cuda" if config["cuda"] else "cpu")

    return config


def training_operations(config):
    """ Operations for the training procedure.
    """

    # Load data
    df = load_data(data_file=config["data_file"])

    # Split data
    split_df = split_data(
        df=df, shuffle=config["shuffle"],
        train_size=config["train_size"],
        val_size=config["val_size"],
        test_size=config["test_size"])

    # Preprocessing
    preprocessed_df = preprocess_data(split_df)

    # Load dataset and vectorizer
    dataset = Dataset.load_dataset_and_make_vectorizer(preprocessed_df)
    dataset.save_vectorizer(config["vectorizer_file"])
    vectorizer = dataset.vectorizer

    # Sample checks
    sample(dataset=dataset)

    # Initializing model
    model = initialize_model(config=config, vectorizer=vectorizer)

    # Training
    trainer = Trainer(
        dataset=dataset, model=model, model_file=config["model_file"],
        save_dir=config["save_dir"], device=config["device"],
        shuffle=config["shuffle"], num_epochs=config["num_epochs"],
        batch_size=config["batch_size"], learning_rate=config["learning_rate"],
        early_stopping_criteria=config["early_stopping_criteria"])
    trainer.run_train_loop()

    # Testing
    y_pred, y_test = trainer.run_test_loop()

    # Save all results
    trainer.save_train_state()


def inference_operations():
    """Inference operations.
    """
    # Experiment id
    experiment_id = "1545431238_932ed400-056f-11e9-82ef-8e0065915101"

    # Load train config
    config_filepath = os.path.join(BASE_DIR, "experiments", experiment_id, "config.json")
    with open(config_filepath, 'r') as fp:
        config = json.load(fp)

    # Load vectorizer
    with open(config["vectorizer_file"]) as fp:
        vectorizer = Vectorizer.from_serializable(json.load(fp))

    # Initializing model
    model = initialize_model(config=config, vectorizer=vectorizer)

    # Load model
    model.load_state_dict(torch.load(config["model_file"]))
    model = model.to("cpu")

    # Inference
    inference = Inference(model=model, vectorizer=vectorizer)
    X = "President Bush signed the peace treaty at the White House dinner."
    top_k = inference.predict_top_k(preprocess_text(X), k=len(vectorizer.y_vocab))
    print ("{}".format(X))
    for result in top_k:
        print ("{} (p={:0.2f})".format(result['y'],
                                       result['probability']))








