import os
from flask import Flask, make_response, jsonify, redirect, session, render_template, url_for

from document_classification.config import cache, DevelopmentConfig, ProductionConfig
from document_classification.api.api import _api

# Define Flask app
application = Flask(__name__)
application.url_map.strict_slashes = False
#cache.init_app(application)

# Choose config
application.config.from_object(DevelopmentConfig)

# Register blueprints
application.register_blueprint(_api)

# 404
@application.errorhandler(404)
def page_not_found(e):
    """Redirect all nonexistent URLS.
    """
    resp = {"response": "404 PAGE NOT FOUND"}
    status = 404
    return make_response(jsonify(resp), status)

# Internal error
@application.errorhandler(500)
def internal_error(error):
    resp = {"response": "500 INTERNAL ERROR"}
    status = 500
    return make_response(jsonify(resp), status)


if __name__ == "__main__":
    application.run(host=application.config['HOST'],
                    port=application.config['PORT'])