import os
from flask import Blueprint, jsonify, make_response, request
import json
import torch
from threading import Thread

from document_classification.api.utils import train, get_experiment_ids, \
    get_experiment_info, delete_experiment

# Define blueprint
_api = Blueprint("_api", __name__)


# Health check
@_api.route("/", methods=["GET"])
#@cache.cached(timeout=3600)
def _health_check():
    """Health check.
    """
    resp = {"response": "We are live!"}
    status = 200
    return make_response(jsonify(resp), status)


# Training
@_api.route("/train", methods=["POST"])
def _train():
    """Training a model.
    """

    if request.method == "POST":

        # Get config filepath
        config_filepath = request.json["config_filepath"]

        # Training
        config = train(config_filepath=config_filepath)

        resp = {
            "experiment_id": config["experiment_id"],
            "save_dir": config["save_dir"],
            }
        status = 200

        return make_response(jsonify(resp), status)


# List of experiments
@_api.route("/experiments", methods=["GET"])
def _experiments():
    """Get a list of available experiments.
    """
    # Get ids
    experiment_ids = get_experiment_ids()
    resp = {"experiment_ids": experiment_ids}
    status = 200
    return make_response(jsonify(resp), status)


# Experiement info
@_api.route("/info/<experiment_id>", methods=["GET"])
def _info(experiment_id):
    """Get experiment info.
    """

    # Latest experiment_id
    if experiment_id == "latest":
        experiment_id = get_experiment_ids()[-1]

    # Get info
    experiment_info = get_experiment_info(experiment_id=experiment_id)
    resp = {"experiment_info": experiment_info}
    status = 200
    return make_response(jsonify(resp), status)

# Delete an experiment
@_api.route("/delete/<experiment_id>", methods=["GET"])
def _delete(experiment_id):
    """Delete an experiment
    """
    # Get ids
    response = delete_experiment(experiment_id=experiment_id)
    resp = {"response": response}
    status = 200
    return make_response(jsonify(resp), status)

if __name__ == '__main__':
    pass
